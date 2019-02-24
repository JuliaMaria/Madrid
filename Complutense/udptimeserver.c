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

  if (argc < 4) {
    printf("ERROR\n");
    return -1;
  }

  struct addrinfo hints;
  struct addrinfo *result1, *result2, *iterator;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_flags = AI_PASSIVE;

  if (getaddrinfo(argv[1], argv[2], &hints, &result1) != 0) {
      printf("ERROR: getaddrinfo\n");
      exit(EXIT_FAILURE);
  }

  if (getaddrinfo(argv[1], argv[3], &hints, &result2) != 0) {
      printf("ERROR: getaddrinfo\n");
      exit(EXIT_FAILURE);
  }

  int socketUDP1 = socket(result1->ai_family, result1->ai_socktype, result1->ai_protocol);
  int socketUDP2 = socket(result2->ai_family, result2->ai_socktype, result2->ai_protocol);

  if (bind(socketUDP1, result1->ai_addr, result1->ai_addrlen) != 0) {
    printf("ERROR: bind\n");
    exit(EXIT_FAILURE);
  }

  if (bind(socketUDP2, result2->ai_addr, result2->ai_addrlen) != 0) {
    printf("ERROR: bind\n");
    exit(EXIT_FAILURE);
  }

  freeaddrinfo(result1);
  freeaddrinfo(result2);

  char buf[256];
  char host[NI_MAXHOST];
  char serv[NI_MAXSERV];

  struct sockaddr_storage client_addr;
  socklen_t client_addrlen = sizeof(client_addr);
  printf("Listening\n");

  fd_set ports;
  FD_ZERO(&ports);
  FD_SET(socketUDP1, &ports);
  FD_SET(socketUDP2, &ports);
  int port;
  int selected;

  while(port != -1){
    
    fd_set ports;
    FD_ZERO(&ports);
    FD_SET(socketUDP1, &ports);
    FD_SET(socketUDP2, &ports);

    port = select((socketUDP1 < socketUDP2) ? socketUDP2 + 1 : socketUDP1 + 1, &ports, NULL, NULL, NULL);

    if (port > 0) {
      if (FD_ISSET(socketUDP1, &ports)) {
        selected = socketUDP1;
      } else if (FD_ISSET(socketUDP2, &ports)) {
        selected = socketUDP2;
        }
    }

    ssize_t bytes = recvfrom(selected, buf, 256, 0, (struct sockaddr *) &client_addr, &client_addrlen);

    getnameinfo((struct sockaddr *) &client_addr, client_addrlen, host, NI_MAXHOST, serv, NI_MAXSERV, NI_NUMERICHOST|NI_NUMERICSERV);

    printf("%s:%s\n", host, serv);

    time_t tiempo = time(NULL);
    struct tm *tm = localtime(&tiempo);
    size_t max;
    char s[50];

    size_t bytesT = strftime(s, max, "%I:%M:%S %p", tm);
    s[bytesT] = '\0';
    sendto(selected, s, bytesT, 0, (struct sockaddr *) &client_addr, client_addrlen);

  }

  return 0;

}
