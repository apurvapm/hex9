#ifndef HEX_SELFPLAY_RECORD_HPP
#define HEX_SELFPLAY_RECORD_HPP

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "hex/board.hpp"

namespace hex {

// Storage format for self-play games.
//
// Positions are stored as move sequences plus per-move visit counts, not as
// encoded tensors. Replaying is cheap and it keeps shards roughly twenty times
// smaller — a 9x9 game holds about seventy positions, and a full tensor per
// position would be 1.3 KB each.
//
// It also means the trainer reaches its inputs through exactly the same replay
// and encode path that the golden fixture in tests/test_encoding.cpp already
// pins. There is no second encoding implementation to drift.
//
// Layout, little-endian throughout:
//
//   magic     4 bytes  "HEX9"
//   version   u16
//   size      u16      board size N
//   games     u32
//   per game:
//     plies   u16
//     winner  i8       +1 red, -1 blue
//     per ply:
//       move      u8   action actually played, N*N is swap
//       entries   u8   number of actions with a non-zero visit count
//       per entry:
//         action  u8
//         visits  u16
struct SelfPlayRecord {
  std::vector<int> moves;
  // visits[t] holds (action, count) pairs for the search at ply t.
  std::vector<std::vector<std::pair<int, std::uint16_t>>> visits;
  int winner = 0;  // +1 red, -1 blue
};

class RecordWriter {
 public:
  static constexpr std::uint16_t kVersion = 1;

  RecordWriter(const std::string& path, int board_size)
      : file_(std::fopen(path.c_str(), "wb")), board_size_(board_size) {
    if (file_ == nullptr) return;
    std::fwrite("HEX9", 1, 4, file_);
    Put16(kVersion);
    Put16(static_cast<std::uint16_t>(board_size));
    // Patched with the real count on close, so a crashed run still leaves a
    // readable prefix rather than a file claiming games it never wrote.
    Put32(0);
  }

  ~RecordWriter() { Close(); }

  bool ok() const { return file_ != nullptr; }

  void Write(const SelfPlayRecord& record) {
    if (file_ == nullptr) return;
    Put16(static_cast<std::uint16_t>(record.moves.size()));
    Put8(static_cast<std::uint8_t>(record.winner > 0 ? 1 : 0xFF));

    for (std::size_t ply = 0; ply < record.moves.size(); ++ply) {
      Put8(static_cast<std::uint8_t>(record.moves[ply]));
      const auto& entries = record.visits[ply];
      Put8(static_cast<std::uint8_t>(entries.size()));
      for (const auto& [action, count] : entries) {
        Put8(static_cast<std::uint8_t>(action));
        Put16(count);
      }
    }
    ++games_;
  }

  void Close() {
    if (file_ == nullptr) return;
    std::fseek(file_, 8, SEEK_SET);
    Put32(games_);
    std::fclose(file_);
    file_ = nullptr;
  }

  std::uint32_t games() const { return games_; }

 private:
  void Put8(std::uint8_t v) { std::fwrite(&v, 1, 1, file_); }
  void Put16(std::uint16_t v) {
    const std::uint8_t bytes[2] = {static_cast<std::uint8_t>(v & 0xFF),
                                   static_cast<std::uint8_t>(v >> 8)};
    std::fwrite(bytes, 1, 2, file_);
  }
  void Put32(std::uint32_t v) {
    const std::uint8_t bytes[4] = {static_cast<std::uint8_t>(v & 0xFF),
                                   static_cast<std::uint8_t>((v >> 8) & 0xFF),
                                   static_cast<std::uint8_t>((v >> 16) & 0xFF),
                                   static_cast<std::uint8_t>(v >> 24)};
    std::fwrite(bytes, 1, 4, file_);
  }

  std::FILE* file_ = nullptr;
  int board_size_ = 0;
  std::uint32_t games_ = 0;
};

class RecordReader {
 public:
  explicit RecordReader(const std::string& path)
      : file_(std::fopen(path.c_str(), "rb")) {
    if (file_ == nullptr) return;
    char magic[4] = {};
    if (std::fread(magic, 1, 4, file_) != 4 ||
        std::string(magic, 4) != "HEX9") {
      Fail();
      return;
    }
    version_ = Get16();
    board_size_ = Get16();
    games_ = Get32();
    if (version_ != RecordWriter::kVersion) Fail();
  }

  ~RecordReader() {
    if (file_ != nullptr) std::fclose(file_);
  }

  bool ok() const { return file_ != nullptr; }
  int board_size() const { return board_size_; }
  std::uint32_t games() const { return games_; }

  bool Read(SelfPlayRecord& record) {
    if (file_ == nullptr || read_ >= games_) return false;
    const std::uint16_t plies = Get16();
    const std::uint8_t winner = Get8();

    record.moves.clear();
    record.visits.clear();
    record.winner = winner == 1 ? 1 : -1;

    for (std::uint16_t ply = 0; ply < plies; ++ply) {
      record.moves.push_back(Get8());
      const std::uint8_t entries = Get8();
      std::vector<std::pair<int, std::uint16_t>> row;
      row.reserve(entries);
      for (std::uint8_t i = 0; i < entries; ++i) {
        const int action = Get8();
        const std::uint16_t count = Get16();
        row.emplace_back(action, count);
      }
      record.visits.push_back(std::move(row));
    }
    ++read_;
    return true;
  }

 private:
  void Fail() {
    std::fclose(file_);
    file_ = nullptr;
  }
  std::uint8_t Get8() {
    std::uint8_t v = 0;
    if (std::fread(&v, 1, 1, file_) != 1) Fail();
    return v;
  }
  std::uint16_t Get16() {
    std::uint8_t b[2] = {};
    if (std::fread(b, 1, 2, file_) != 2) Fail();
    return static_cast<std::uint16_t>(b[0] | (b[1] << 8));
  }
  std::uint32_t Get32() {
    std::uint8_t b[4] = {};
    if (std::fread(b, 1, 4, file_) != 4) Fail();
    return static_cast<std::uint32_t>(b[0]) |
           (static_cast<std::uint32_t>(b[1]) << 8) |
           (static_cast<std::uint32_t>(b[2]) << 16) |
           (static_cast<std::uint32_t>(b[3]) << 24);
  }

  std::FILE* file_ = nullptr;
  std::uint16_t version_ = 0;
  int board_size_ = 0;
  std::uint32_t games_ = 0;
  std::uint32_t read_ = 0;
};

}  // namespace hex

#endif  // HEX_SELFPLAY_RECORD_HPP
