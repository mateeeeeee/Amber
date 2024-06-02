#include "ConsoleVariable.h"
#include "ConsoleManager.h"

namespace amber
{
	IConsoleVariable::IConsoleVariable(char const* name) : name(name)
	{
		ConsoleManager::RegisterConsoleVariable(this, name);
	}

}

