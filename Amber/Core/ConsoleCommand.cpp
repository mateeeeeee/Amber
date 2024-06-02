#include "ConsoleCommand.h"
#include "ConsoleManager.h"

namespace amber
{
	IConsoleCommand::IConsoleCommand(char const* name) : name(name)
	{
		ConsoleManager::RegisterConsoleCommand(this, name);
	}
}

