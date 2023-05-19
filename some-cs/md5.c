#include <stdio.h>
#include <string.h>
//#define OPENSSL_API_COMPAT 10000
//#define OPENSSL_API_LEVEL 10000
#include <openssl/ssl.h>

int main() {
  // Create an MD5 hash object.
  MD5_CTX ctx;
  MD5_Init(&ctx);

  // Update the state of the MD5 hash object with the data.
  MD5_Update(&ctx, "Hello, World!", strlen("Hello, World!"));

  // Calculate the MD5 hash of the data.
  unsigned char hash[MD5_DIGEST_LENGTH];
  MD5_Final(hash, &ctx);

  // Print the MD5 hash.
  for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
    printf("%02x", hash[i]);
  }
  printf("\n");

  return 0;
}

