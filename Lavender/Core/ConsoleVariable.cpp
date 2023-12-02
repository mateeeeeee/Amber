#include "ConsoleVariable.h"
#include "ConsoleManager.h"

namespace lavender
{
	IConsoleVariable::IConsoleVariable(char const* name) : name(name)
	{
		ConsoleManager::RegisterConsoleVariable(this, name);
	}

}

