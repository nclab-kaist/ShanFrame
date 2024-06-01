def generate_intrin() -> str:
    return """
#ifndef SHANFRAME_INTRIN_H
#define SHANFRAME_INTRIN_H
#include <stdint.h>
#include <string.h>

#define MAX(a, b) ((a) < (b))? (b) : (a)
#define MIN(a, b) ((a) < (b))? (a) : (b)

__attribute__((always_inline)) static __inline  int32_t read_int8x4(const int8_t *src) {
    int32_t result;
    memcpy(&result, src, 4);
    return result;
}

__attribute__((always_inline)) static __inline  int32_t read_int8x4_ia(const int8_t **src) {
    int32_t result;
    memcpy(&result, *src, 4);
    *src += 4;
    return result;
}

__attribute__((always_inline)) static __inline  uint32_t __SMLAD (uint32_t op1, uint32_t op2, uint32_t op3){
  uint32_t result;

  __asm__ volatile ("smlad %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}

__attribute__((always_inline)) static __inline  uint32_t __SMLABB(uint32_t op1, uint32_t op2, uint32_t op3) {
    uint32_t result;

    __asm__ volatile ("smlabb %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3));
    return(result);
}

__attribute__((always_inline)) static __inline  uint32_t __SMLATB(uint32_t op1, uint32_t op2, uint32_t op3) {
    uint32_t result;

    __asm__ volatile ("smlatb %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3));
    return(result);
}

__attribute__((always_inline)) static __inline  uint32_t __SMLABT(uint32_t op1, uint32_t op2, uint32_t op3) {
    uint32_t result;

    __asm__ volatile ("smlabt %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3));
    return(result);
}

__attribute__((always_inline)) static __inline  uint32_t __SMLATT(uint32_t op1, uint32_t op2, uint32_t op3) {
    uint32_t result;

    __asm__ volatile ("smlatt %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3));
    return(result);
}

__attribute__((always_inline)) static __inline  uint32_t __SXTB16(uint32_t op1){
    uint32_t result;

    __asm__ ("sxtb16 %0, %1" : "=r" (result) : "r" (op1));
    return(result);
}

__attribute__((always_inline)) static __inline  uint32_t __SXTB16_RORn(uint32_t op1, const int rotate){
    uint32_t result;

    __asm__ volatile ("sxtb16 %0, %1, ROR %2" : "=r" (result) : "r" (op1), "i" (rotate) );
    return(result);
}
#endif
"""
