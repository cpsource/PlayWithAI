#include <stdlib.h>

struct node {
  void *data;
  struct node *next;
};

struct queue {
  struct node *head;
  struct node *tail;
};

void insque(struct queue *q, void *data) {
  struct node *new_node = malloc(sizeof(struct node));
  new_node->data = data;
  new_node->next = NULL;

  if (q->head == NULL) {
    q->head = new_node;
    q->tail = new_node;
  } else {
    q->tail->next = new_node;
    q->tail = new_node;
  }
}

void remque(struct queue *q) {
  struct node *current = q->head;
  q->head = current->next;

  if (q->head == NULL) {
    q->tail = NULL;
  }

  free(current);
}

int main() {
  struct queue *q = malloc(sizeof(struct queue));
  q->head = NULL;
  q->tail = NULL;

  // Test 1: Insert three elements into the queue
  insque(q, "Hello");
  insque(q, "World");
  insque(q, "!");

  // Test 2: Print the first and last element of the queue
  printf("%s\n", q->head->data); // Hello
  printf("%s\n", q->tail->data); // !

  // Test 3: Remove the first element of the queue
  remque(q);

  // Test 4: Print the first element of the queue
  printf("%s\n", q->head->data); // World

  // Test 5: Free the queue
  while (q->head != NULL) {
    struct node *current = q->head;
    q->head = current->next;
    free(current);
  }

  free(q);

  return 0;
}

