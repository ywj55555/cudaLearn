#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h> 
#include <math.h>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#define BIG (1e7)
// #define DEBUG
using namespace std;
template<typename theIterator> void print(theIterator begin, theIterator end);


template<typename T> __global__ void
mergeVec_half(T *A, T *tmp, const int64_t vSize) {

    /* splict the vector A into two halfs
     * merge these two half together
     *
     * tmp is a temporary vector to 
     * receive the merge result
     */

    int64_t left = blockIdx.x * vSize;
    int64_t right = left + vSize - 1;
    int64_t mid = (left + right) / 2;

    int64_t i = left, j = mid + 1, k = left;  // index of left half, right half, and the mergeVec
    while ((i <= mid) && (j <= right)) {
        if (A[i] <= A[j]) {
            tmp[k] = A[i];
            ++i; ++k;
        } else {
            tmp[k] = A[j];
            ++j; ++k;
        }
    }
    if (i > mid) {
        for (; j <= right; ++j, ++k) {
            tmp[k] = A[j];
        }
    } else {
        for (; i <= mid; ++i, ++k) {
            tmp[k] = A[i];
        }
    }
    /// copy tmp to A
    for (k = left; k <= right; ++k) {
        A[k] = tmp[k];
    }
}


template<typename theIterator, typename T> void 
mergeSort_power2n(theIterator begin, theIterator end, T args) {
    /* 
        sort a vector with size of power(2, n)
    */
    clock_t begT, endT;

    T *dataA, *dataTmp;
    int64_t vSize = end - begin;
    cudaMalloc((void**)&dataA, sizeof(*begin) * vSize);
    cudaMalloc((void**)&dataTmp, sizeof(*begin) * vSize);

    #ifdef DEBUG
    int64_t n = 0;
    if (vSize >= 2) {
        for (int64_t i = 1; i < vSize; i <<= 1) {
            n += 1;
        }
    } else {
        return;
    }
    /// check whether n is correct
    if (((int64_t)1 << n) > vSize) {
        cerr << "\033[31;1m error! vSize != 2 ** n \033[0m";
        exit(-1);
    }
    #endif

    begT = clock();
    cudaMemcpy(dataA, &(*begin), sizeof(*begin) * vSize, cudaMemcpyHostToDevice);

    /// merge hierarchically
    for (int64_t i = 2; i <= vSize; i <<= 1) {  // i is the size of vector
        mergeVec_half <<<vSize / i, 1>>> (dataA, dataTmp, i);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            // Possibly: exit(-1) if program cannot continue....
        } 
        // cudaDeviceSynchronize();  // no need to synchronize here, because kernel2 will wait for kernel1
        #ifdef DEBUG
            cudaMemcpy(&(*begin), dataA, sizeof(*begin) * vSize, cudaMemcpyDeviceToHost);
            cout << "merging Vector, vec = ";
            print(begin, end);
        #endif
    }
    /// data from device to host
    cudaMemcpy(&(*begin), dataA, sizeof(*begin) * vSize, cudaMemcpyDeviceToHost);
    endT = clock();
    cout << "inside GPU operation, time = " << endT - begT << endl;

    cudaFree(dataA);
    cudaFree(dataTmp);
}
template<typename theIterator> inline void 
mergeSort_power2n(theIterator begin, theIterator end) {
    mergeSort_power2n(begin, end, *begin);
}


template<typename theIterator, typename T> void
mergeVec(
    theIterator beg1, theIterator end1,
    theIterator beg2, theIterator end2,
    T args
) {
    /* 
     * merge 2 vectors with arbitrary length
     * of each vector
     */
    vector<T> tmp((end1 - beg1) + (end2 - beg2));
    theIterator i = beg1, j = beg2;
    theIterator k = tmp.begin();

    while(i != end1 && j != end2) {
        if (*i <= *j) {
            *k = *i;
            ++i; ++k;
        } else {
            *k = *j;
            ++j; ++k;
        }
    }
    if (i == end1) {
        while (j != end2) {
            *k = *j;
            ++j; ++k;
        }
    } else {
        while (i != end1) {
            *k = *i;
            ++i; ++k;
        }
    }
    /// copy tmp to original vectors
    k = tmp.begin();
    for (i = beg1; i != end1; ++i, ++k) {
        *i = *k;
    }
    for (j = beg2; j != end2; ++j, ++k) {
        *j = *k;
    }
}
template<typename theIterator> inline void 
mergeVec(theIterator beg1, theIterator end1, theIterator beg2, theIterator end2) {
    mergeVec(beg1, end1, beg2, end2, *beg1);
}


template<typename vec> void 
mergeSort_gpu(vec &A) {
    /* can deal with arbitary size of vector */
    vector<bool> binA;
    int64_t vSize = A.size(), n = A.size();
    int64_t one = 1;
    while (n > 0) {
        if (n & one) {
            binA.push_back(true);
        } else {
            binA.push_back(false);
        }
        n >>= 1;
    }

    vector<int64_t> idxVec;
    idxVec.push_back(0);
    for (int64_t i = 0; i != binA.size(); ++i) {
        if (binA[i]) {
            idxVec.push_back(idxVec.back() + (one << i));
        }
    }

    for (int64_t i = 0; i != idxVec.size() - 1; ++i) {
        mergeSort_power2n(A.begin() + idxVec[i], A.begin() + idxVec[i + 1]);
    }
    /// merge all ranges of vector
    for (int64_t i = 1; i != idxVec.size() - 1; ++i) {
        mergeVec(
            A.begin(), A.begin() + idxVec[i],
            A.begin() + idxVec[i], A.begin() + idxVec[i + 1]
        );
    }
}


template<typename theIterator, typename T> void 
mergeSort_cpu(theIterator begin, theIterator end, T args) {

    /* cpu version of the merge sort */

    if (end - 1 - begin < 1) return;

    vector<T> tmp(end - begin, 0);

    theIterator left = begin, right = end - 1;
    theIterator mid = left + (right - left) / 2;

    mergeSort_cpu(begin, mid + 1, args);
    mergeSort_cpu(mid + 1, end, args);

    theIterator i = begin;
    theIterator j = mid + 1;
    theIterator k = tmp.begin();
    
    while(i <= mid && j < end) {
        if (*i <= *j) {
            *k = *i;
            ++i; ++k;
        } else {
            *k = *j;
            ++j; ++k;
        }
    }
    if (i > mid) {
        for (; j < end; ++j, ++k) {
            *k = *j;
        }
    } else {
        for (; i <= mid; ++i, ++k) {
            *k = *i;
        }
    }
    for (i = begin, k = tmp.begin(); i != end; ++i, ++k) {
        *i = *k;
    }
}
template<typename theIterator> inline void 
mergeSort_cpu(theIterator begin, theIterator end) {
    mergeSort_cpu(begin, end, *begin);
}


template<typename theIterator> void 
print(theIterator begin, theIterator end) {
    int64_t showNums = 10;
    if (end - begin <= showNums) {
        for (theIterator i = begin; i != end; ++i) {
            cout << *i << ", ";
        } cout << endl;
    } else {
        for (theIterator i = begin; i != begin + showNums / 2; ++i) {
            cout << *i << ", ";
        } cout << "......, ";
        for (theIterator i = end - showNums / 2; i != end; ++i) {
            cout << *i << ", ";
        } cout << endl;
    }
}


int main() {

    clock_t start, end;

    // vector<double> A(pow(2, 20) * 16), B(pow(2, 20) * 16);
    // vector<double> A(19), B(19);
    vector<long long> A(BIG), B(BIG), C(BIG);
    for (int64_t i = A.size() - 1; i != -1; --i) {
        // A[i] = A.size() - 1 - i;
        A[i] = rand();
        C[i] = B[i] = A[i];
    }

    cout << "initially, A = ";
    print(A.begin(), A.end());

    start = clock();  // begin cuda computation
    mergeSort_gpu(A);
    end = clock();  // end cuda computation
    cout << "using GPU, consuming time = " << (end - start) * 1000. / CLOCKS_PER_SEC << " ms" << endl;
    cout << "after sort, A = ";
    print(A.begin(), A.end());

    /// use cpu to sort
    start = clock();
    mergeSort_cpu(B.begin(), B.end());
    end = clock();
    cout << "using CPU, consuming time = " << (end - start) * 1000. / CLOCKS_PER_SEC << " ms" << endl;
    cout << "after sort, B = ";
    print(B.begin(), B.end());

    /// use sort algorithm of stl
    start = clock();
    stable_sort(C.begin(), C.end());
    end = clock();
    cout << "using CPU, stl::stable_sort, consuming time = " << (end - start) * 1000. / CLOCKS_PER_SEC << " ms" << endl;
    cout << "after sort, C = ";
    print(C.begin(), C.end());

    /// test whether A equals C
    bool equal = true;
    for (int64_t i = 0; i != A.size(); ++i) {
        if (A[i] != C[i]) {
            equal = false;
            break;
        }
    }
    if (!equal) {
        cerr << "\033[31;1m there is a bug in the program. A != C \033[0m" << endl;
    } else {
        cout << "\033[32;1m very good, A == C \033[0m" << endl;
    }
}
