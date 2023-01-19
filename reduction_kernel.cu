#define BLOCK_SIZE 1024

/* Reduction Kernel --------------------------------------
*       Find local min hash and corresponding nonce values.
*/
__global__
void reduction_kernel(unsigned int* local_hash_array, unsigned int* local_nonce_array, unsigned int* hash_array, unsigned int* nonce_array, unsigned int array_size) {

    unsigned int index = 2 * blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ unsigned int reduction_hash[BLOCK_SIZE];
    __shared__ unsigned int reduction_nonce[BLOCK_SIZE];

    if (index < array_size){
        reduction_hash[threadIdx.x] = hash_array[index];
        reduction_nonce[threadIdx.x] = nonce_array[index];
    }
    else{
        reduction_hash[threadIdx.x] = UINT_MAX;
        reduction_nonce[threadIdx.x] = UINT_MAX;
    }

    if (index + BLOCK_SIZE < array_size){
        if (reduction_hash[threadIdx.x] > hash_array[index + BLOCK_SIZE]){
            reduction_hash[threadIdx.x] = hash_array[index + BLOCK_SIZE];
            reduction_nonce[threadIdx.x] = nonce_array[index + BLOCK_SIZE];
        }
    }

    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride = stride / 2){
        __syncthreads();
        if(threadIdx.x < stride){
            if(reduction_hash[threadIdx.x] > reduction_hash[threadIdx.x + stride]){
                reduction_hash[threadIdx.x] = reduction_hash[threadIdx.x + stride];
                reduction_nonce[threadIdx.x] = reduction_nonce[threadIdx.x + stride];
            }
        }
    }

    if (threadIdx.x == 0){
        local_hash_array[blockIdx.x] = reduction_hash[0];
        local_nonce_array[blockIdx.x] = reduction_nonce[0];
    }



} // End Reduction Kernel //