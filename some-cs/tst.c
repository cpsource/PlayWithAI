#include <stdio.h>
#include <limits.h>

void tst(void)
{unsigned int x = -1;
if ( /* x < 0 && */ x <= UINT_MAX) {
  printf("Hello World\n");
}
return;
}

int main() {
  unsigned int x = -30;

  // Check if x is less than 0.
  if (x < 0) {
    printf("x is less than 0\n");
  } else {
    printf("x is not less than 0\n");
  }

  // Check if x is equal to 0.
  if (x == 0) {
    printf("x is equal to 0\n");
  } else {
    printf("x is not equal to 0\n");
  }

  // Check if x is greater than 0.
  if (x > 0) {
    printf("x is greater than 0\n");
  } else {
    printf("x is not greater than 0\n");
  }

  return 0;
}
