#include <stdio.h>

int main() {
  int num = 1234567890;
  int new_num = (num >> 3) & ((1 << 8) - 1);
  printf("from 0x%08x, The new number is %d 0x%08x\n", num,new_num,new_num);
  return 0;
}
