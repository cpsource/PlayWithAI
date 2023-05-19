#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <crypt.h>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s filename\n", argv[0]);
    exit(1);
  }

  FILE *fp = fopen(argv[1], "rb");
  if (fp == NULL) {
    fprintf(stderr, "Error opening file: %s\n", argv[1]);
    exit(1);
  }

  char md5sum[33];
  MD5_CTX ctx;
  MD5Init(&ctx);

  char buf[1024];
  size_t bytes_read;
  while ((bytes_read = fread(buf, 1, sizeof(buf), fp)) > 0) {
    MD5Update(&ctx, buf, bytes_read);
  }

  MD5Final(md5sum, &ctx);

  printf("MD5 sum of %s: %s\n", argv[1], md5sum);

  fclose(fp);

  return 0;
}

