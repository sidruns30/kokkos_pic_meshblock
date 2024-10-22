#ifndef PARTICLE_TAGS_HPP
#define PARTICLE_TAGS_HPP

#include <Kokkos_Core.hpp>

struct PrtlSendTag {
  inline static constexpr short im1_jm1_km1 { 2 };
  inline static constexpr short im1_jm1__k0 { 3 };
  inline static constexpr short im1_jm1_kp1 { 4 };
  inline static constexpr short im1__j0_km1 { 5 };
  inline static constexpr short im1__j0__k0 { 6 };
  inline static constexpr short im1__j0_kp1 { 7 };
  inline static constexpr short im1_jp1_km1 { 8 };
  inline static constexpr short im1_jp1__k0 { 9 };
  inline static constexpr short im1_jp1_kp1 { 10 };
  inline static constexpr short i0__jm1_km1 { 11 };
  inline static constexpr short i0__jm1__k0 { 12 };
  inline static constexpr short i0__jm1_kp1 { 13 };
  inline static constexpr short i0___j0_km1 { 14 };
  inline static constexpr short i0___j0_kp1 { 15 };
  inline static constexpr short i0__jp1_km1 { 16 };
  inline static constexpr short i0__jp1__k0 { 17 };
  inline static constexpr short i0__jp1_kp1 { 18 };
  inline static constexpr short ip1_jm1_km1 { 19 };
  inline static constexpr short ip1_jm1__k0 { 20 };
  inline static constexpr short ip1_jm1_kp1 { 21 };
  inline static constexpr short ip1__j0_km1 { 22 };
  inline static constexpr short ip1__j0__k0 { 23 };
  inline static constexpr short ip1__j0_kp1 { 24 };
  inline static constexpr short ip1_jp1_km1 { 25 };
  inline static constexpr short ip1_jp1__k0 { 26 };
  inline static constexpr short ip1_jp1_kp1 { 27 };
};

KOKKOS_INLINE_FUNCTION auto
  SendTag(short tag, bool im1, bool ip1, bool jm1, bool jp1, bool km1, bool kp1)
    -> short {
  return ((im1 && jm1 && km1) * (PrtlSendTag::im1_jm1_km1 - 1) +
          (im1 && jm1 && kp1) * (PrtlSendTag::im1_jm1_kp1 - 1) +
          (im1 && jp1 && km1) * (PrtlSendTag::im1_jp1_km1 - 1) +
          (im1 && jp1 && kp1) * (PrtlSendTag::im1_jp1_kp1 - 1) +
          (ip1 && jm1 && km1) * (PrtlSendTag::ip1_jm1_km1 - 1) +
          (ip1 && jm1 && kp1) * (PrtlSendTag::ip1_jm1_kp1 - 1) +
          (ip1 && jp1 && km1) * (PrtlSendTag::ip1_jp1_km1 - 1) +
          (ip1 && jp1 && kp1) * (PrtlSendTag::ip1_jp1_kp1 - 1) +
          (im1 && jm1 && !km1 && !kp1) * (PrtlSendTag::im1_jm1__k0 - 1) +
          (im1 && jp1 && !km1 && !kp1) * (PrtlSendTag::im1_jp1__k0 - 1) +
          (ip1 && jm1 && !km1 && !kp1) * (PrtlSendTag::ip1_jm1__k0 - 1) +
          (ip1 && jp1 && !km1 && !kp1) * (PrtlSendTag::ip1_jp1__k0 - 1) +
          (im1 && !jm1 && !jp1 && km1) * (PrtlSendTag::im1__j0_km1 - 1) +
          (im1 && !jm1 && !jp1 && kp1) * (PrtlSendTag::im1__j0_kp1 - 1) +
          (ip1 && !jm1 && !jp1 && km1) * (PrtlSendTag::ip1__j0_km1 - 1) +
          (ip1 && !jm1 && !jp1 && kp1) * (PrtlSendTag::ip1__j0_kp1 - 1) +
          (!im1 && !ip1 && jm1 && km1) * (PrtlSendTag::i0__jm1_km1 - 1) +
          (!im1 && !ip1 && jm1 && kp1) * (PrtlSendTag::i0__jm1_kp1 - 1) +
          (!im1 && !ip1 && jp1 && km1) * (PrtlSendTag::i0__jp1_km1 - 1) +
          (!im1 && !ip1 && jp1 && kp1) * (PrtlSendTag::i0__jp1_kp1 - 1) +
          (!im1 && !ip1 && !jm1 && !jp1 && km1) * (PrtlSendTag::i0___j0_km1 - 1) +
          (!im1 && !ip1 && !jm1 && !jp1 && kp1) * (PrtlSendTag::i0___j0_kp1 - 1) +
          (!im1 && !ip1 && jm1 && !km1 && !kp1) * (PrtlSendTag::i0__jm1__k0 - 1) +
          (!im1 && !ip1 && jp1 && !km1 && !kp1) * (PrtlSendTag::i0__jp1__k0 - 1) +
          (im1 && !jm1 && !jp1 && !km1 && !kp1) * (PrtlSendTag::im1__j0__k0 - 1) +
          (ip1 && !jm1 && !jp1 && !km1 && !kp1) * (PrtlSendTag::ip1__j0__k0 - 1) +
          1) *
         tag;
}

#endif
