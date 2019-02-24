#include <stdio.h>
#include <signal.h>
#include <stdlib.h>

int main(int argc, char **argv) {

  if (argc != 2) {
    printf("ERROR\n");
    return -1;
  }

  sigset_t set;

  sigemptyset(&set);
  sigaddset(&set, SIGINT);
  sigaddset(&set, SIGTSTP);

  sigprocmask(SIG_BLOCK, &set, NULL);

  char *sleep_secs = argv[1];
  int secs = atoi(sleep_secs);
  printf("%d s\n", secs);
  sleep(secs);

  sigset_t pending;
  sigpending(&pending);

  if (sigismember(&pending, SIGINT) == 1) {
    printf("SIGINT\n");
    sigdelset(&set, SIGINT);
  } else {
    printf("No SIGINT\n");
  }

  if (sigismember(&pending, SIGTSTP) == 1) {
    printf("SIGTSTP\n");
  } else {
    printf("No SIGTSTP\n");
  }

  sigprocmask(SIG_UNBLOCK, &set, NULL);
  printf("Finalized\n");

  return 0;

}


