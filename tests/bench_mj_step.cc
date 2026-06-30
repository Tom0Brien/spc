#include <mujoco/mujoco.h>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <vector>
#include <cstring>

int main() {
    mjModel* model = mj_loadXML("models/franka_push/scene.xml", nullptr, nullptr, 0);
    if (!model) {
        std::cerr << "Could not load model\n";
        return 1;
    }
    
    std::cout << "Threads: " << omp_get_max_threads() << "\n";
    
    int num_samples = 128;
    std::vector<mjData*> datas(num_samples);
    for (int i = 0; i < num_samples; ++i) datas[i] = mj_makeData(model);
    
    mjData* root_data = mj_makeData(model);
    
    int horizon = 25;
    int substeps = 4;
    int num_iters = 2;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    bool printed[128] = {false};
    
    omp_set_num_threads(8);
    
    for (int iter = 0; iter < num_iters; ++iter) {
        #pragma omp parallel for
        for (int i = 0; i < num_samples; ++i) {
            if (iter == 0 && !printed[omp_get_thread_num()]) {
                printed[omp_get_thread_num()] = true;
                printf("Thread %d running\n", omp_get_thread_num());
            }
            std::memcpy(datas[i]->qpos, root_data->qpos, model->nq * sizeof(mjtNum));
            std::memcpy(datas[i]->qvel, root_data->qvel, model->nv * sizeof(mjtNum));
            std::memcpy(datas[i]->ctrl, root_data->ctrl, model->nu * sizeof(mjtNum));
            std::memcpy(datas[i]->act, root_data->act, model->na * sizeof(mjtNum));
            std::memcpy(datas[i]->mocap_pos, root_data->mocap_pos, 3 * model->nmocap * sizeof(mjtNum));
            std::memcpy(datas[i]->mocap_quat, root_data->mocap_quat, 4 * model->nmocap * sizeof(mjtNum));
            std::memcpy(datas[i]->qacc_warmstart, root_data->qacc_warmstart, model->nv * sizeof(mjtNum));
            
            mj_kinematics(model, datas[i]);
            for (int h = 0; h < horizon; ++h) {
                // mock apply control
                for(int j=0; j<7; ++j) datas[i]->ctrl[j] = 0.0f;
                datas[i]->ctrl[7] = 0.82f;
                for (int s = 0; s < substeps; ++s) {
                    mj_step(model, datas[i]);
                }
            }
        }
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    std::cout << "Time taken: " << ms << " ms\n";
    std::cout << "Steps per second: " << (num_samples * horizon * substeps * num_iters) / (ms / 1000.0) << "\n";
    
    return 0;
}
