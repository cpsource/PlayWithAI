#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <search.h>

struct element {
  struct element *forward;
  struct element *backward;
  char *name;
};

static struct element *
new_element(void)
{
  struct element *e = malloc(sizeof(*e));
  if (e == NULL) {
    fprintf(stderr, "malloc() failed\n");
    exit(EXIT_FAILURE);
  }

  return e;
}

void
insque(void *e, void *p)
{
  struct element *elem = (struct element *)e;
  struct element *prev = (struct element *)p;

  if (prev == NULL) {
    elem->forward = elem;
    elem->backward = elem;
  } else {
    elem->forward = prev->forward;
    prev->forward = elem;
    elem->backward = prev;
  }
}

#if 0
int
main(int argc, char *argv[])
{
  struct element *first, *elem, *prev;
  int circular, opt, errfnd;

  /* The "-c" command-line option can be used to specify that the
     list is circular. */

  errfnd = 0;
  circular = 0;
  while ((opt = getopt(argc, argv, "c")) != -1) {
    switch (opt) {
    case 'c':
      circular = 1;
      break;
    default:
      errfnd = 1;
      break;
    }
  }

  if (errfnd || optind >= argc) {
    fprintf(stderr,  "Usage: %s [-c] string...\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  /* Create first element and place it in the linked list. */

  elem = new_element();
  first = elem;

  elem->name = argv[optind];

  if (circular) {
    insque(elem, elem);
  } else {
    insque(elem, NULL);
  }

  /* Add remaining command-line arguments as list elements. */

  for (; optind < argc; optind++) {
    prev = elem;

    elem = new_element();
    elem->name = argv[optind];
    insque(elem, prev);
  }

  /* Traverse the list from the start, printing element names. */

  printf("Traversing completed list:\n");
  elem = first;
  do {
    printf("    %s\n", elem->name);
    elem = elem->forward;
  } while (elem != NULL && elem != first);

  if (elem == first)
    printf("That was a circular list\n");

  exit(EXIT_SUCCESS);
}
#endif

int
test_main(int argc, char *argv[])
{
  int i, n;
  struct element *list;
  struct element *elem;
  
  /* Create a linked list */
  list = NULL;
  for (i = 1; i < argc; i++) {
    elem = new_element();
    elem->name = argv[i];
    insque(elem, list);
    list = elem;
  }

  /* Traverse the list and print the element names */
  printf("Traversing the linked list:\n");
  for (elem = list; elem != NULL; elem = elem->forward) {
    printf("    %s\n", elem->name);
  }

  /* Check the list for errors */
  n = 0;
  for (elem = list; elem != NULL; elem = elem->forward) {
    n++;
  }
  if (n != argc - 1) {
    fprintf(stderr, "Error: Linked list has wrong number of elements\n");
    return 1;
  }

  /* Free the memory allocated for the linked list */
  while (list != NULL) {
    struct element *next = list->forward;
    free(list);
    list = next;
  }

  return 0;
}

int main(int argc, char *argv[])
{
  test_main(argc,argv);
  return 0;
}
