#ifndef MY_COMMUNICATOR_H
#define MY_COMMUNICATOR_H

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <arpa/inet.h>

#include <linux/if_ether.h>
#include <linux/in.h>


#define TENSOR_NUM 128

struct packet_t {
    __u32 worker_bitmap;
    __u32 aggregator_index;
    __u32 gradient_index;
	__u32 gradient[TENSOR_NUM];
} __attribute__((packed));

void send_gradients(__u32 *gradient_array,int packet_num, __u32 dst_ip, int worker_id, __u32 aggregator_index, int tensor_index);

#endif