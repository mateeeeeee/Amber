#include "Lavender.h"
#include "Core/Window.h"
#include "Editor/Editor.h"

int main(int argc, char* argv[])
{
	lavender::Initialize();
	{
		lavender::Window window(1080, 720, "lavender");
		lavender::Editor editor(window);

		while (window.Loop())
		{
			editor.Run();
		}
	}
	lavender::Destroy();
	return 0;
}