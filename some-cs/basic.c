#include <openssl/conf.h>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <stdio.h>
#include <string.h>

int encrypt(unsigned char *plaintext, 
            int plaintext_len,
            unsigned char *key,
            unsigned char *iv,
            unsigned char *ciphertext);

int decrypt(unsigned char *ciphertext, 
            int ciphertext_len,
            unsigned char *key,
            unsigned char *iv,
            unsigned char *plaintext);

void handleErrors(void);

int main(int argc, char *argv[]) { 
  /* Load the human readable error strings for libcrypto */
  ERR_load_crypto_strings();

  /* Load all digest and cipher algorithms */
  OpenSSL_add_all_algorithms();

  /* Load config file, and other important initialisation */
  //OPENSSL_config(NULL);
  OPENSSL_no_config();

  unsigned char *key = (unsigned char *)"01234567890123456789012345678901";
  unsigned char *iv = (unsigned char *)"someIV";
  unsigned char *plaintext = (unsigned char *) "Bajja";
  unsigned char ciphertext[128];
  unsigned char decryptedtext[128];

  int decryptedtext_len, ciphertext_len;
  ciphertext_len = encrypt(plaintext, strlen ((char *)plaintext), key, iv, ciphertext);
  printf("Ciphertext is:\n");
  BIO_dump_fp(stdout, (const char *)ciphertext, ciphertext_len);

  decryptedtext_len = decrypt(ciphertext, ciphertext_len, key, iv, decryptedtext);
  decryptedtext[decryptedtext_len] = '\0';
  printf("Decrypted text is:\n");
  printf("%s\n", decryptedtext);

  /* Removes all digests and ciphers */
  EVP_cleanup();

  /* if you omit the next, a small leak may be left when you make use of the BIO (low level API) for e.g. base64 transformations */
  CRYPTO_cleanup_all_ex_data();

  /* Remove error strings */
  ERR_free_strings();

  return 0;
}

int encrypt(unsigned char* plaintext,
            int plaintext_len,
            unsigned char* key,
            unsigned char* iv,
            unsigned char* ciphertext) {
  EVP_CIPHER_CTX* ctx;
  int len;
  int ciphertext_len;

  /* Create and initialise the context */
  if(!(ctx = EVP_CIPHER_CTX_new())) {
    handleErrors();
  }

  /* Initialise the encryption operation. IMPORTANT - ensure you use a key
   * and IV size appropriate for your cipher
   * In this example we are using 256 bit AES (i.e. a 256 bit key). The
   * IV size for *most* modes is the same as the block size. For AES this
   * is 128 bits */
  if(1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv)) {
    handleErrors();
  }

  /* Provide the message to be encrypted, and obtain the encrypted output.
   * EVP_EncryptUpdate can be called multiple times if necessary
   */
  if(1 != EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, plaintext_len)) {
    handleErrors();
  }
  ciphertext_len = len;

  /* Finalise the encryption. Further ciphertext bytes may be written at
   * this stage.
   */
  if(1 != EVP_EncryptFinal_ex(ctx, ciphertext + len, &len)) {
    handleErrors();
  }
  ciphertext_len += len;

  /* Clean up */
  EVP_CIPHER_CTX_free(ctx);
  return ciphertext_len;
}

int decrypt(unsigned char* ciphertext,
            int ciphertext_len,
            unsigned char* key,
            unsigned char* iv,
            unsigned char* plaintext) {
  EVP_CIPHER_CTX* ctx;
  int len;
  int plaintext_len;

  /* Create and initialise the context */
  if(!(ctx = EVP_CIPHER_CTX_new())) {
    handleErrors();
  }

  /* Initialise the decryption operation. IMPORTANT - ensure you use a key
   * and IV size appropriate for your cipher
   * In this example we are using 256 bit AES (i.e. a 256 bit key). The
   * IV size for *most* modes is the same as the block size. For AES this
   * is 128 bits */
  if(1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv)) {
    handleErrors();
  }

  /* Provide the message to be decrypted, and obtain the plaintext output.
   * EVP_DecryptUpdate can be called multiple times if necessary
   */
  if(1 != EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len)) {
    handleErrors();
  }
  plaintext_len = len;

  /* Finalise the decryption. Further plaintext bytes may be written at
   * this stage.
   */
  if(1 != EVP_DecryptFinal_ex(ctx, plaintext + len, &len)) {
    handleErrors();
  }
  plaintext_len += len;

  /* Clean up */
  EVP_CIPHER_CTX_free(ctx);
  return plaintext_len;
}

void handleErrors(void) {
  ERR_print_errors_fp(stderr);
  abort();
}

