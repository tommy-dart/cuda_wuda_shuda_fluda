// IMPORTS OF NOTE
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

namespace name
{
    string title="fluid_dynamics";
    string team="cuda_wuda_shuda";
    string author_1="tommy_white";
    string author_2="mon_rozbeer";
}




void fluflu_sim() {

}


void run_da_waves() {
    uint howmanywaves = 2;
    string howbig = "6ft";
    string[] period = ["27s", "24s"];
    uint[] dir = [270, 285];

    // run it
    return;
}


int main() {

    string fname = "schmonika.da_bish";
    out.open(fname.c_str());

    if (out.fail()) {
        printf("\n\nUR FILE (%s) is BROKE YO", fname.c_str());
        return 1;
    }

    run_da_waves();
    // fluflu_sim(); wtf was this called for

    return 0;
}
