#include <bard/bard.h>

int main() {
  // Create a new Bard client.
  bard_client *client = bard_client_new();

  // Set the client's API key.
  bard_client_set_api_key(client, "YOUR_API_KEY");

  // Ask Bard a question.
  bard_response *response = bard_client_ask(client, "What is the capital of France?");

  // Print the response.
  printf("The capital of France is %s.\n", bard_response_get_text(response));

  // Free the response.
  bard_response_free(response);

  // Free the client.
  bard_client_free(client);

  return 0;
}

