// Minimal SHA-256 (FIPS 180-4), harness-side only.
//
// Used to record and enforce the digest of the fixture file a report was
// produced from, so that a result computed from a stale input cannot be
// mistaken for a valid one. No cryptographic claim is made beyond "this is
// SHA-256"; correctness is checked against the system sha256sum in the
// conformance tests.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace astra::harness {

namespace detail {

constexpr uint32_t kRoundConstants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u};

[[nodiscard]] constexpr uint32_t rotr(uint32_t x, int n) noexcept {
  return (x >> n) | (x << (32 - n));
}

}  // namespace detail

[[nodiscard]] inline std::string sha256_hex(std::string_view data) {
  uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

  std::vector<uint8_t> message(data.begin(), data.end());
  const uint64_t bit_length = static_cast<uint64_t>(message.size()) * 8u;

  message.push_back(0x80u);
  while (message.size() % 64u != 56u) {
    message.push_back(0x00u);
  }
  for (int shift = 7; shift >= 0; --shift) {
    message.push_back(static_cast<uint8_t>((bit_length >> (8 * shift)) & 0xffu));
  }

  for (size_t offset = 0; offset < message.size(); offset += 64u) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      const size_t base = offset + static_cast<size_t>(4 * i);
      w[i] = (static_cast<uint32_t>(message[base]) << 24) |
             (static_cast<uint32_t>(message[base + 1]) << 16) |
             (static_cast<uint32_t>(message[base + 2]) << 8) |
             static_cast<uint32_t>(message[base + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = detail::rotr(w[i - 15], 7) ^ detail::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = detail::rotr(w[i - 2], 17) ^ detail::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    for (int i = 0; i < 64; ++i) {
      const uint32_t big_s1 = detail::rotr(e, 6) ^ detail::rotr(e, 11) ^ detail::rotr(e, 25);
      const uint32_t choose = (e & f) ^ (~e & g);
      const uint32_t t1 = h + big_s1 + choose + detail::kRoundConstants[i] + w[i];
      const uint32_t big_s0 = detail::rotr(a, 2) ^ detail::rotr(a, 13) ^ detail::rotr(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = big_s0 + majority;

      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (uint32_t word : state) {
    for (int shift = 28; shift >= 0; shift -= 4) {
      out.push_back(kHex[(word >> shift) & 0xfu]);
    }
  }
  return out;
}

}  // namespace astra::harness
