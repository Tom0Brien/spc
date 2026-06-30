#include <mujoco/mujoco.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <cstdarg>
#include <cstdio>

#include "spc/algs/cem.h"
#include "spc/algs/optimizer.h"
#include "spc/core/policy.h"
#include "spc/core/task.h"
#include "spc/core/task_factory.h"

namespace py = pybind11;

// Since passing mjModel/mjData pointers between Python's mujoco package and C++ is tricky
// (though possible via ctypes), we create a lightweight C++ environment that owns
// its own C++ mjModel and mjData. We sync states back to Python for rendering.
class SpcEnv {
public:
    SpcEnv(const std::string& xml_path) : owns_data_(true) {
        char error[1000];
        m_ = mj_loadXML(xml_path.c_str(), nullptr, error, 1000);
        if (!m_) {
            throw std::runtime_error(std::string("Failed to load model: ") + error);
        }
        d_ = mj_makeData(m_);
        if (m_->nkey > 0) {
            mj_resetDataKeyframe(m_, d_, 0);
        }
    }

    SpcEnv(uintptr_t model_ptr, uintptr_t data_ptr) : owns_data_(false) {
        m_ = reinterpret_cast<mjModel*>(model_ptr);
        d_ = reinterpret_cast<mjData*>(data_ptr);
    }

    ~SpcEnv() {
        if (owns_data_) {
            if (d_)
                mj_deleteData(d_);
            if (m_)
                mj_deleteModel(m_);
        }
    }

    mjModel* GetModel() { return m_; }
    mjData* GetData() { return d_; }

    void StepMPC(std::shared_ptr<spc::algs::Optimizer> optimizer, int sim_steps_per_replan) {
        int ctrl_dim = optimizer->GetControlDim();
        std::vector<float> best_action(ctrl_dim);
        optimizer->Optimize(d_, best_action.data());

        auto task = optimizer->GetTask();
        if (task) {
            task->ApplyControl(m_, d_, best_action.data());
        } else {
            for (int i = 0; i < m_->nu && i < ctrl_dim; ++i) {
                d_->ctrl[i] = best_action[i];
            }
        }

        for (int step = 0; step < sim_steps_per_replan; ++step) {
            mj_step(m_, d_);
        }
    }

    void SetMocapPos(int mocap_id, py::array_t<double> pos) {
        auto r = pos.unchecked<1>();
        for (int i = 0; i < 3 && i < r.shape(0); ++i) {
            d_->mocap_pos[3 * mocap_id + i] = r(i);
        }
    }

    void SetMocapQuat(int mocap_id, py::array_t<double> quat) {
        auto r = quat.unchecked<1>();
        for (int i = 0; i < 4 && i < r.shape(0); ++i) {
            d_->mocap_quat[4 * mocap_id + i] = r(i);
        }
    }

    void SetQpos(py::array_t<double> qpos) {
        auto r = qpos.unchecked<1>();
        for (int i = 0; i < m_->nq && i < r.shape(0); ++i) {
            d_->qpos[i] = r(i);
        }
    }

    void SetQvel(py::array_t<double> qvel) {
        auto r = qvel.unchecked<1>();
        for (int i = 0; i < m_->nv && i < r.shape(0); ++i) {
            d_->qvel[i] = r(i);
        }
    }

    void SetCtrl(py::array_t<double> ctrl) {
        auto r = ctrl.unchecked<1>();
        for (int i = 0; i < m_->nu && i < r.shape(0); ++i) {
            d_->ctrl[i] = r(i);
        }
    }

    void Forward() { mj_forward(m_, d_); }

    py::array_t<double> GetQpos() { return py::array_t<double>(m_->nq, d_->qpos); }

    int GetNu() { return m_->nu; }

private:
    mjModel* m_;
    mjData* d_;
    bool owns_data_;
};

#include "spc/core/mlp_policy.h"
#include "spc/core/onnx_policy.h"

std::shared_ptr<spc::algs::Optimizer> MakeCEM(SpcEnv& env, std::shared_ptr<spc::core::Task> task,
                                              std::shared_ptr<spc::core::Policy> policy,
                                              const spc::algs::CEMConfig& config) {
    return std::make_shared<spc::algs::CEM>(env.GetModel(), task, policy, config);
}

PYBIND11_MODULE(spc_py, m) {
    py::class_<SpcEnv>(m, "SpcEnv")
        .def(py::init<const std::string&>())
        .def(py::init<uintptr_t, uintptr_t>())
        .def("step_mpc", &SpcEnv::StepMPC)
        .def("set_mocap_pos", &SpcEnv::SetMocapPos)
        .def("set_mocap_quat", &SpcEnv::SetMocapQuat)
        .def("set_qpos", &SpcEnv::SetQpos)
        .def("set_qvel", &SpcEnv::SetQvel)
        .def("set_ctrl", &SpcEnv::SetCtrl)
        .def("forward", &SpcEnv::Forward)
        .def("get_qpos", &SpcEnv::GetQpos)
        .def_property_readonly("nu", &SpcEnv::GetNu);

    py::class_<spc::core::Task, std::shared_ptr<spc::core::Task>>(m, "Task")
        .def("get_observation",
             [](std::shared_ptr<spc::core::Task> task, SpcEnv& env, int obs_dim) {
                 std::vector<float> obs(obs_dim, 0.0f);
                 task->GetObservation(env.GetModel(), env.GetData(), obs.data());
                 return py::array_t<float>(obs.size(), obs.data());
             })
        .def("running_cost", [](std::shared_ptr<spc::core::Task> task, SpcEnv& env, py::array_t<float> control) {
            auto r = control.unchecked<1>();
            std::vector<float> ctrl(r.shape(0));
            for (int i = 0; i < r.shape(0); ++i)
                ctrl[i] = r(i);
            return task->RunningCost(env.GetModel(), env.GetData(), ctrl.data());
        });

    m.def(
        "create_task",
        [](const std::string& name, SpcEnv& env, py::dict num_params, py::dict str_params) {
            spc::core::TaskConfig config;
            for (auto item : num_params) {
                config.numeric_params[py::cast<std::string>(item.first)] = py::cast<double>(item.second);
            }
            for (auto item : str_params) {
                config.string_params[py::cast<std::string>(item.first)] = py::cast<std::string>(item.second);
            }
            return spc::core::TaskFactory::GetInstance().Create(name, env.GetModel(), config);
        },
        "Create a registered task by name", py::arg("name"), py::arg("env"), py::arg("numeric_params") = py::dict(),
        py::arg("string_params") = py::dict());

    py::class_<spc::core::Policy, std::shared_ptr<spc::core::Policy>>(m, "Policy");

    py::class_<spc::core::ONNXPolicy, spc::core::Policy, std::shared_ptr<spc::core::ONNXPolicy>>(m, "ONNXPolicy")
        .def(py::init<const std::string&>());

    py::class_<spc::core::MLPPolicy, spc::core::Policy, std::shared_ptr<spc::core::MLPPolicy>>(m, "MLPPolicy")
        .def(py::init<const std::string&>());

    py::class_<spc::algs::CEMConfig>(m, "CEMConfig")
        .def(py::init<>())
        .def_readwrite("num_samples", &spc::algs::CEMConfig::num_samples)
        .def_readwrite("num_elites", &spc::algs::CEMConfig::num_elites)
        .def_readwrite("num_knots", &spc::algs::CEMConfig::num_knots)
        .def_readwrite("num_iterations", &spc::algs::CEMConfig::num_iterations)
        .def_readwrite("plan_horizon_steps", &spc::algs::CEMConfig::plan_horizon_steps)
        .def_readwrite("sim_substeps", &spc::algs::CEMConfig::sim_substeps)
        .def_readwrite("control_dim", &spc::algs::CEMConfig::control_dim)
        .def_readwrite("obs_dim", &spc::algs::CEMConfig::obs_dim)
        .def_readwrite("sigma_init", &spc::algs::CEMConfig::sigma_init)
        .def_readwrite("sigma_min", &spc::algs::CEMConfig::sigma_min)
        .def_readwrite("explore_fraction", &spc::algs::CEMConfig::explore_fraction)
        .def_readwrite("num_threads", &spc::algs::CEMConfig::num_threads);

    py::class_<spc::algs::Optimizer, std::shared_ptr<spc::algs::Optimizer>>(m, "Optimizer");

    m.def("CEM", &MakeCEM, "Create a CEM optimizer", py::arg("env"), py::arg("task"), py::arg("policy"),
          py::arg("config"));
}
