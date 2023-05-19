#include <stdio.h>
#include <stdlib.h>

void *get_file_contents(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  long size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  void *buffer = malloc(size);
  if (buffer == NULL) {
    fclose(fp);
    return NULL;
  }

  fread(buffer, 1, size, fp);
  fclose(fp);

  return buffer;
}

