#pragma once

namespace amber
{
	class IImGuiManager
	{
	public:
		virtual ~IImGuiManager() = default;

		virtual void Initialize() = 0;
		virtual void Shutdown() = 0;
		virtual void BeginFrame() = 0;
		virtual void EndFrame() = 0;
		virtual void Render() = 0;
	};
}
