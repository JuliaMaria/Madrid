#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

int main(){

  char *tuberia1 = "pipe1";
  mkfifo(tuberia1, 0644);

  char *tuberia2 = "pipe2";
  mkfifo(tuberia2, 0644);

  char buffer[256];

  int pipe1 = open(tuberia1, O_RDONLY | O_NONBLOCK);
  int pipe2 = open(tuberia2, O_RDONLY | O_NONBLOCK);

  int cambios, pipeactual;

  while (cambios != -1) {
    fd_set set;
    FD_ZERO(&set);
    FD_SET(pipe1, &set);
    FD_SET(pipe2, &set);
    int numpipeactual;

    cambios = select((pipe1 < pipe2) ? pipe2 + 1 : pipe1 + 1, &set, NULL, NULL, NULL);

    if (cambios > 0){
      if (FD_ISSET(pipe1, &set)) {
        numpipeactual = 1;
        pipeactual = pipe1;
      } else if (FD_ISSET(pipe2, &set)) {
        numpipeactual = 2;
        pipeactual = pipe2;
      }

      ssize_t readsize = 255;
      while (readsize == 255) {
	readsize = read(pipeactual, buffer, 255);
	buffer[readsize] = '\0';
	printf("Pipe %i: %s", numpipeactual, buffer);
      }

      if (readsize != 255 && numpipeactual == 1) {
        close(pipe1);
        pipe1 = open(tuberia1, O_RDONLY | O_NONBLOCK);
	if (pipe1 == -1) {
	  perror("Unable to open the second pipe");
	  close(pipe1);
	  close(pipe2);
	  return -1;
	}
      } else if(readsize != 255 && numpipeactual == 2) {
          close(pipe2);
          pipe2 = open(tuberia2, O_RDONLY | O_NONBLOCK);
	  if (pipe2 == -1) {
	    perror("Unable to open the second pipe");
	    close(pipe1);
	    close(pipe2);
	    return -1;
	  }
      }
    }

  }

  close(pipe1);
  close(pipe2);

  return 0;

}
