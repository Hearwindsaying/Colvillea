#pragma once

namespace colvillea
{
namespace core
{
// TODO: tmps
struct Camera
{
    vec3f m_camera_pos;
    vec3f m_camera_d00;
    vec3f m_camera_ddu;
    vec3f m_camera_ddv;

    bool operator==(const Camera& rhs) const
    {
        return this->m_camera_pos == rhs.m_camera_pos &&
            this->m_camera_d00 == rhs.m_camera_d00 &&
            this->m_camera_ddu == rhs.m_camera_ddu &&
            this->m_camera_ddv == rhs.m_camera_ddv;
    }

    bool operator!=(const Camera& rhs) const
    {
        return !(*this == rhs);
    }
};
} // namespace core
} // namespace colvillea