#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <mujoco/mujoco.h>
#include "spc/core/task.h"

namespace spc {
namespace core {

class TaskFactory {
public:
    using CreatorFunc = std::function<std::shared_ptr<Task>(mjModel*, const TaskConfig&)>;

    static TaskFactory& GetInstance();

    void Register(const std::string& name, CreatorFunc func) {
        registry_[name] = std::move(func);
    }

    std::shared_ptr<Task> Create(const std::string& name, mjModel* model, const TaskConfig& config) {
        auto it = registry_.find(name);
        if (it != registry_.end()) {
            return it->second(model, config);
        }
        throw std::invalid_argument("Task not registered: " + name);
    }

private:
    TaskFactory() = default;
    std::unordered_map<std::string, CreatorFunc> registry_;
};

#define REGISTER_TASK(Name, Class, ID) \
    class ID##Registrar { \
    public: \
        ID##Registrar() { \
            spc::core::TaskFactory::GetInstance().Register(Name, [](mjModel* model, const spc::core::TaskConfig& config) { \
                return std::make_shared<Class>(model, config); \
            }); \
        } \
    }; \
    static ID##Registrar global_##ID##_registrar;

} // namespace core
} // namespace spc
