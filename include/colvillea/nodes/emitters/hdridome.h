#pragma once

#include <librender/nodebase/emitter.h>
#include <librender/nodebase/texture.h>

namespace colvillea
{
namespace core
{
class HDRIDome : public Emitter
{
public:
    HDRIDome(Scene* pScene, const std::shared_ptr<Texture>& envtex) :
        Emitter{pScene, kernel::EmitterType::HDRIDome},
        m_envmap{envtex}
    {}

    const std::shared_ptr<Texture>& getEnvmap() const noexcept
    {
        return this->m_envmap;
    }

    ~HDRIDome() {}

private:
    std::shared_ptr<Texture> m_envmap;

};
} // namespace core
} // namespace colvillea
