#include "spc/core/task_factory.h"

namespace spc {
namespace core {

TaskFactory& TaskFactory::GetInstance() {
    static TaskFactory instance;
    return instance;
}

}  // namespace core
}  // namespace spc
