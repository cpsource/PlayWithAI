#include <openssl/evp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  // Check the number of arguments.
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <file>\n", argv[0]);
    return 1;
  }

  // Open the file.
  FILE *file = fopen(argv[1], "rb");
  if (file == NULL) {
    fprintf(stderr, "Could not open file: %s\n", argv[1]);
    return 1;
  }

  // Create an MD5 hash object.
  EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
  if (mdctx == NULL) {
    fprintf(stderr, "Could not create MD5 hash object\n");
    return 1;
  }

  // Initialize the hash object.
  if (!EVP_DigestInit_CTX(mdctx, EVP_md5())) {
    fprintf(stderr, "Could not initialize MD5 hash object\n");
    return 1;
  }

  // Update the hash object with the file contents.
  char buffer[1024];
  size_t bytes_read;
  while ((bytes_read = fread(buffer, 1, sizeof(buffer), file)) > 0) {
    EVP_DigestUpdate(mdctx, buffer, bytes_read);
  }

  // Calculate the MD5 hash.
  unsigned char hash[EVP_MD_CTX_size(mdctx)];
  if (!EVP_DigestFinal_CTX(mdctx, hash, NULL)) {
    fprintf(stderr, "Could not calculate MD5 hash\n");
    return 1;
  }

  // Print the MD5 hash.
  for (int i = 0; i < EVP_MD_CTX_size(mdctx); i++) {
    printf("%02x", hash[i]);
  }
  printf("\n");

  // Close the file.
  fclose(file);

  // Destroy the hash object.
  EVP_MD_CTX_free(mdctx);

  return 0;
}

