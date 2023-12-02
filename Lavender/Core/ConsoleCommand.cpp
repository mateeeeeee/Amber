#include "ConsoleCommand.h"
#include "ConsoleManager.h"

namespace lavender
{
	IConsoleCommand::IConsoleCommand(char const* name) : name(name)
	{
		ConsoleManager::RegisterConsoleCommand(this, name);
	}
}

