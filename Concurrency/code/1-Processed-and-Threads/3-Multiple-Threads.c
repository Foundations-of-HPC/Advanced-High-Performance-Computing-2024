#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

void *thread_function(void *unused) {
  (void)unused;

  printf("PID: %ld, thread_id: %d\n", (long)getpid(), gettid());
  sleep(120);

  return NULL;
}

int main() {
  pthread_t thread;

  if (pthread_create(&thread, NULL, thread_function, NULL)) {
    fprintf(stderr, "pthread_create failure.\n");
    return EXIT_FAILURE;
  }

  thread_function(NULL);

  if (pthread_join(thread, NULL)) {
    fprintf(stderr, "pthread_joint failure.\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
