#include "spc/tasks/particle.h"
#include <stdexcept>

namespace spc {
namespace tasks {

Particle::Particle(mjModel* model, const spc::core::TaskConfig& config) {
    if (!model) {
        throw std::invalid_argument("Model cannot be null");
    }
    pointmass_site_id_ = mj_name2id(model, mjOBJ_SITE, "pointmass");
    if (pointmass_site_id_ < 0) {
        throw std::runtime_error("Could not find site 'pointmass' in model");
    }
    nu_ = model->nu;
}

void Particle::GetObservation(const mjModel* model, const mjData* data, float* obs_out) const {
    // For now, the particle task observation logic isn't heavily specified
    // since we use it as a purely analytical baseline. We can fill qpos/qvel if needed.
}

double Particle::TerminalCost(const mjModel* model, const mjData* data) const {
    // position_cost = sum((state.site_xpos[pointmass_id] - state.mocap_pos[0])^2)
    double pos_cost = 0.0;
    const double* site_xpos = data->site_xpos + 3 * pointmass_site_id_;
    const double* mocap_pos = data->mocap_pos; // first mocap body
    
    for (int i = 0; i < 3; ++i) {
        double diff = site_xpos[i] - mocap_pos[i];
        pos_cost += diff * diff;
    }
    
    // velocity_cost = sum(qvel^2)
    double vel_cost = 0.0;
    for (int i = 0; i < model->nv; ++i) {
        vel_cost += data->qvel[i] * data->qvel[i];
    }
    
    return 5.0 * pos_cost + 0.1 * vel_cost;
}

double Particle::RunningCost(const mjModel* model, const mjData* data, const float* control) const {
    double state_cost = TerminalCost(model, data);
    
    double control_cost = 0.0;
    for (int i = 0; i < nu_; ++i) {
        control_cost += control[i] * control[i];
    }
    
    return state_cost + 0.1 * control_cost;
}

} // namespace tasks
} // namespace spc

#include "spc/core/task_factory.h"
REGISTER_TASK("ParticleTask", spc::tasks::Particle, ParticleTask)
