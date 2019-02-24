#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <sys/types.h> 
#include <sys/socket.h> 
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <time.h>

int main (int argc, char**argv) {

  if (argc < 3) {
    printf("ERROR\n");
    return -1;
  }

  struct addrinfo hints;
  struct addrinfo *result1, *iterator;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_flags = AI_PASSIVE;

  if (getaddrinfo(argv[1], argv[2], &hints, &result1) != 0) {
      printf("ERROR: getaddrinfo\n");
      exit(EXIT_FAILURE);
  }

  int socketUDP1 = socket(result1->ai_family, result1->ai_socktype, result1->ai_protocol);

  if (bind(socketUDP1, result1->ai_addr, result1->ai_addrlen) != 0) {
    printf("ERROR: bind\n");
    exit(EXIT_FAILURE);
  }

  freeaddrinfo(result1);

  char buf[256];
  char host[NI_MAXHOST];
  char serv[NI_MAXSERV];

  struct sockaddr_storage client_addr;
  socklen_t client_addrlen = sizeof(client_addr);
  printf("Listening\n");

  while(1){

    ssize_t bytes = recvfrom(socketUDP1, buf, 256, 0, (struct sockaddr *) &client_addr, &client_addrlen);

    getnameinfo((struct sockaddr *) &client_addr, client_addrlen, host, NI_MAXHOST, serv, NI_MAXSERV, NI_NUMERICHOST|NI_NUMERICSERV);

    char s[50];
    int bytesT = sprintf(s, "%s:%s", host, serv);
    s[bytesT] = '\0';
    sendto(socketUDP1, s, bytesT, 0, (struct sockaddr *) &client_addr, client_addrlen);

  }

  return 0;

}



