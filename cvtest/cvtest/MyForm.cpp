#include "MyForm.h"
#include "stdafx.h"


using namespace System;
using namespace System::Windows::Forms;

[STAThread]
//HACK: when switching back to the mainprogramme test, remember to check Project Properties > Linker > Advanced > Entry Point and delete Main
int WinMain(){
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);

	cvtest::MyForm form;
	Application::Run(%form);
}
