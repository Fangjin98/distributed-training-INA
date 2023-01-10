#include "communicator.h"

void send_gradients(__u32 *gradient_array, int packet_num, __u32 dst_ip, int worker_id, __u32 aggregator_index, int tensor_index) {
    int socket_fd;
	struct sockaddr_in sock_send;
    memset(&sock_send, 0, sizeof(struct sockaddr_in));
    sock_send.sin_family = AF_INET;
	sock_send.sin_addr.s_addr = htonl(dst_ip);

	if ((socket_fd = socket(AF_INET, SOCK_RAW, IPPROTO_UDP)) < 0) {
	    perror("ERROR: Failed to create raw socket.\n");
		exit(-1);
	}

    int send_buff_size = 4096 * 4096;
    setsockopt(socket_fd,SOL_SOCKET,SO_SNDBUF,&send_buff_size,sizeof(send_buff_size));

    int bitmap = 1 << (worker_id-1);
    
    clock_t start, end;
    start=clock();

    for (int i=0; i<packet_num; i++){
        struct packet_t packet;

        packet.worker_bitmap = htonl(bitmap);
        packet.aggregator_index= htonl(aggregator_index);
        packet.gradient_index = htonl(tensor_index + i);
        memcpy(packet.gradient, gradient_array + i * TENSOR_NUM, TENSOR_NUM * sizeof(__u32));

        // for endian conversion
        for (int j=0; j<TENSOR_NUM ; j++)
        {
            packet.gradient[j] = htonl(packet.gradient[j]);
        }
        
	    if(sendto(socket_fd, &packet, sizeof(struct packet_t), 0, (struct sockaddr *)&sock_send, sizeof(struct sockaddr_in)) < 0){
            perror("ERROR: Failed to call sendto()");
		    exit(-1);
        }
    }

    end = clock();
    printf("Time consumed: %lf s\n", (double)(end - start) / CLOCKS_PER_SEC);

	return;
}