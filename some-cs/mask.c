#include <stdio.h>
#include <stdint.h>

unsigned int get_bits(unsigned int value, int start, int end) {
  // Check if the start and end bits are valid
  if (start < 0 || start > 31 || end < 0 || end > 31) {
    return 0;
  }

  // Mask the bits we want to keep
  unsigned int mask = (1 << (end - start + 1)) - 1;
  printf("mask = 0x%08x\n",mask);

  mask <<= start;

  printf("mask = 0x%08x\n",mask);

  // Shift the value to the correct position
  value >>= start;

  // Return the masked value
  return value & mask;
}

int main(int argc, char *argv[])
{
  printf("0x%08x\n",11);
  printf("%d\n",get_bits(11,3,11));
  return 0;

}
