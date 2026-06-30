#include "spc/tasks/particle.h"
#include <stdexcept>

namespace spc {
namespace tasks {

Particle::Particle(mjModel* model, const spc::core::TaskConfig& config) {
    if (!model) {
        throw std::invalid_argument("Model cannot be null");
    }
    std::string site_name = config.string_params.count("site_name") ? config.string_params.at("site_name") : "pointmass";
    pointmass_site_id_ = mj_name2id(model, mjOBJ_SITE, site_name.c_str());
    if (pointmass_site_id_ < 0) {
        throw std::runtime_error("Could not find site '" + site_name + "' in model");
    }
    
    pos_weight_ = config.numeric_params.count("pos_weight") ? config.numeric_params.at("pos_weight") : 5.0;
    vel_weight_ = config.numeric_params.count("vel_weight") ? config.numeric_params.at("vel_weight") : 0.1;
    ctrl_weight_ = config.numeric_params.count("ctrl_weight") ? config.numeric_params.at("ctrl_weight") : 0.1;
    nu_ = model->nu;
}

void Particle::GetObservation(const mjModel* model, const mjData* data, float* obs_out) const {
    // Currently, particle task doesn't use observations for control in these examples
    // Left empty for purely analytical testing
}

double Particle::TerminalCost(const mjModel* model, const mjData* data) const {
    double pos_cost = 0.0;
    const double* site_xpos = data->site_xpos + 3 * pointmass_site_id_;
    const double* mocap_pos = data->mocap_pos; // first mocap body
    
    for (int i = 0; i < 3; ++i) {
        double diff = site_xpos[i] - mocap_pos[i];
        pos_cost += diff * diff;
    }
    
    double vel_cost = 0.0;
    for (int i = 0; i < model->nv; ++i) {
        vel_cost += data->qvel[i] * data->qvel[i];
    }
    
    return pos_weight_ * pos_cost + vel_weight_ * vel_cost;
}

double Particle::RunningCost(const mjModel* model, const mjData* data, const float* control) const {
    double state_cost = TerminalCost(model, data);
    
    double control_cost = 0.0;
    for (int i = 0; i < nu_; ++i) {
        control_cost += control[i] * control[i];
    }
    
    return state_cost + ctrl_weight_ * control_cost;
}

} // namespace tasks
} // namespace spc

#include "spc/core/task_factory.h"
REGISTER_TASK("ParticleTask", spc::tasks::Particle, ParticleTask)
