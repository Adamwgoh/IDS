#pragma once

#include <opencv\cv.h>
#include <opencv\highgui.h>

namespace cvtest {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for MyForm
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
		}
	//status bools
	private: bool isCStreamOn;
	private: bool isDStreamOn;
	private: bool isDeviceConnected;

	//entities
	private: System::Windows::Forms::Button^  device_button;
	private: System::Windows::Forms::Label^  Ldevice_status;
	private: System::Windows::Forms::Label^  device_status;
	private: System::Windows::Forms::Button^  cstream_button;
	private: System::Windows::Forms::Button^  dstream_button;
	

	private: Microsoft::VisualBasic::PowerPacks::ShapeContainer^  shapeContainer1;
	private: Microsoft::VisualBasic::PowerPacks::OvalShape^  c_status;
	private: Microsoft::VisualBasic::PowerPacks::OvalShape^  d_status;
	private: System::Windows::Forms::TabControl^  MainStreamTab;
	private: System::Windows::Forms::TabPage^  cstream_box;
	private: System::Windows::Forms::TabPage^  dstream_box;
	private: System::Windows::Forms::StatusStrip^  stat_strip;
	private: System::Windows::Forms::Button^  saveFrame_button;
	private: System::Windows::Forms::Button^  savePCD_button;

	private: System::Windows::Forms::SaveFileDialog^  saveFrame_dialog;
	private: System::Windows::Forms::SaveFileDialog^  savePCD_dialog;







	protected: 

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->device_button = (gcnew System::Windows::Forms::Button());
			this->Ldevice_status = (gcnew System::Windows::Forms::Label());
			this->device_status = (gcnew System::Windows::Forms::Label());
			this->cstream_button = (gcnew System::Windows::Forms::Button());
			this->dstream_button = (gcnew System::Windows::Forms::Button());
			this->shapeContainer1 = (gcnew Microsoft::VisualBasic::PowerPacks::ShapeContainer());
			this->d_status = (gcnew Microsoft::VisualBasic::PowerPacks::OvalShape());
			this->c_status = (gcnew Microsoft::VisualBasic::PowerPacks::OvalShape());
			this->MainStreamTab = (gcnew System::Windows::Forms::TabControl());
			this->cstream_box = (gcnew System::Windows::Forms::TabPage());
			this->dstream_box = (gcnew System::Windows::Forms::TabPage());
			this->stat_strip = (gcnew System::Windows::Forms::StatusStrip());
			this->saveFrame_button = (gcnew System::Windows::Forms::Button());
			this->savePCD_button = (gcnew System::Windows::Forms::Button());
			this->saveFrame_dialog = (gcnew System::Windows::Forms::SaveFileDialog());
			this->savePCD_dialog = (gcnew System::Windows::Forms::SaveFileDialog());
			this->MainStreamTab->SuspendLayout();
			this->SuspendLayout();
			// 
			// device_button
			// 
			this->device_button->BackColor = System::Drawing::SystemColors::ButtonFace;
			this->device_button->Location = System::Drawing::Point(12, 62);
			this->device_button->Name = L"device_button";
			this->device_button->Size = System::Drawing::Size(129, 39);
			this->device_button->TabIndex = 0;
			this->device_button->Text = L"Start Device";
			this->device_button->UseVisualStyleBackColor = false;
			this->device_button->Click += gcnew System::EventHandler(this, &MyForm::device_buttonClick);
			// 
			// Ldevice_status
			// 
			this->Ldevice_status->AutoSize = true;
			this->Ldevice_status->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->Ldevice_status->Location = System::Drawing::Point(32, 122);
			this->Ldevice_status->Name = L"Ldevice_status";
			this->Ldevice_status->Size = System::Drawing::Size(91, 16);
			this->Ldevice_status->TabIndex = 1;
			this->Ldevice_status->Text = L"Device Status\r\n";
			this->Ldevice_status->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// device_status
			// 
			this->device_status->AutoSize = true;
			this->device_status->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
				static_cast<System::Byte>(0)));
			this->device_status->ForeColor = System::Drawing::Color::OrangeRed;
			this->device_status->Location = System::Drawing::Point(45, 155);
			this->device_status->Name = L"device_status";
			this->device_status->Size = System::Drawing::Size(55, 20);
			this->device_status->TabIndex = 2;
			this->device_status->Text = L"Offline";
			// 
			// cstream_button
			// 
			this->cstream_button->Location = System::Drawing::Point(12, 243);
			this->cstream_button->Name = L"cstream_button";
			this->cstream_button->Size = System::Drawing::Size(88, 30);
			this->cstream_button->TabIndex = 3;
			this->cstream_button->Text = L"Color Stream";
			this->cstream_button->UseVisualStyleBackColor = true;
			this->cstream_button->Click += gcnew System::EventHandler(this, &MyForm::cstream_buttonClick);
			// 
			// dstream_button
			// 
			this->dstream_button->Location = System::Drawing::Point(12, 291);
			this->dstream_button->Name = L"dstream_button";
			this->dstream_button->Size = System::Drawing::Size(88, 28);
			this->dstream_button->TabIndex = 4;
			this->dstream_button->Text = L"Depth Stream";
			this->dstream_button->UseVisualStyleBackColor = true;
			this->dstream_button->Click += gcnew System::EventHandler(this, &MyForm::dstream_buttonClick);
			// 
			// shapeContainer1
			// 
			this->shapeContainer1->Location = System::Drawing::Point(0, 0);
			this->shapeContainer1->Margin = System::Windows::Forms::Padding(0);
			this->shapeContainer1->Name = L"shapeContainer1";
			this->shapeContainer1->Shapes->AddRange(gcnew cli::array< Microsoft::VisualBasic::PowerPacks::Shape^  >(2) {this->d_status, 
				this->c_status});
			this->shapeContainer1->Size = System::Drawing::Size(784, 449);
			this->shapeContainer1->TabIndex = 5;
			this->shapeContainer1->TabStop = false;
			// 
			// d_status
			// 
			this->d_status->BackColor = System::Drawing::Color::White;
			this->d_status->BackStyle = Microsoft::VisualBasic::PowerPacks::BackStyle::Opaque;
			this->d_status->FillColor = System::Drawing::Color::Maroon;
			this->d_status->FillGradientColor = System::Drawing::Color::Maroon;
			this->d_status->FillStyle = Microsoft::VisualBasic::PowerPacks::FillStyle::Solid;
			this->d_status->Location = System::Drawing::Point(110, 299);
			this->d_status->Name = L"d_status";
			this->d_status->Size = System::Drawing::Size(12, 12);
			// 
			// c_status
			// 
			this->c_status->BackColor = System::Drawing::Color::White;
			this->c_status->BackStyle = Microsoft::VisualBasic::PowerPacks::BackStyle::Opaque;
			this->c_status->FillColor = System::Drawing::Color::Maroon;
			this->c_status->FillGradientColor = System::Drawing::Color::Maroon;
			this->c_status->FillStyle = Microsoft::VisualBasic::PowerPacks::FillStyle::Solid;
			this->c_status->Location = System::Drawing::Point(110, 250);
			this->c_status->Name = L"c_status";
			this->c_status->Size = System::Drawing::Size(12, 12);
			// 
			// MainStreamTab
			// 
			this->MainStreamTab->Controls->Add(this->cstream_box);
			this->MainStreamTab->Controls->Add(this->dstream_box);
			this->MainStreamTab->Location = System::Drawing::Point(160, 12);
			this->MainStreamTab->Name = L"MainStreamTab";
			this->MainStreamTab->SelectedIndex = 0;
			this->MainStreamTab->Size = System::Drawing::Size(457, 386);
			this->MainStreamTab->TabIndex = 6;
			// 
			// cstream_box
			// 
			this->cstream_box->Location = System::Drawing::Point(4, 22);
			this->cstream_box->Name = L"cstream_box";
			this->cstream_box->Padding = System::Windows::Forms::Padding(3);
			this->cstream_box->Size = System::Drawing::Size(449, 360);
			this->cstream_box->TabIndex = 0;
			this->cstream_box->Text = L"Color Stream";
			this->cstream_box->UseVisualStyleBackColor = true;
			// 
			// dstream_box
			// 
			this->dstream_box->Location = System::Drawing::Point(4, 22);
			this->dstream_box->Name = L"dstream_box";
			this->dstream_box->Padding = System::Windows::Forms::Padding(3);
			this->dstream_box->Size = System::Drawing::Size(449, 360);
			this->dstream_box->TabIndex = 1;
			this->dstream_box->Text = L"Depth Stream";
			this->dstream_box->UseVisualStyleBackColor = true;
			// 
			// stat_strip
			// 
			this->stat_strip->Location = System::Drawing::Point(0, 427);
			this->stat_strip->Name = L"stat_strip";
			this->stat_strip->Size = System::Drawing::Size(784, 22);
			this->stat_strip->TabIndex = 7;
			this->stat_strip->Text = L"LOADING";
			this->stat_strip->ItemClicked += gcnew System::Windows::Forms::ToolStripItemClickedEventHandler(this, &MyForm::statusStrip1_ItemClicked);
			// 
			// saveFrame_button
			// 
			this->saveFrame_button->Location = System::Drawing::Point(653, 34);
			this->saveFrame_button->Name = L"saveFrame_button";
			this->saveFrame_button->Size = System::Drawing::Size(96, 34);
			this->saveFrame_button->TabIndex = 8;
			this->saveFrame_button->Text = L"Save Frame";
			this->saveFrame_button->UseVisualStyleBackColor = true;
			this->saveFrame_button->Click += gcnew System::EventHandler(this, &MyForm::saveframe_buttonClick);
			// 
			// savePCD_button
			// 
			this->savePCD_button->Location = System::Drawing::Point(653, 74);
			this->savePCD_button->Name = L"savePCD_button";
			this->savePCD_button->Size = System::Drawing::Size(96, 35);
			this->savePCD_button->TabIndex = 9;
			this->savePCD_button->Text = L"Save PCD File";
			this->savePCD_button->UseMnemonic = false;
			this->savePCD_button->UseVisualStyleBackColor = true;
			this->savePCD_button->Click += gcnew System::EventHandler(this, &MyForm::savePCD_buttonClick);
			
			// 
			// saveFrame_dialog
			// 
			this->saveFrame_dialog->DefaultExt = L"jpg";
			this->saveFrame_dialog->Title = L"Save Frame As";
			// 
			// savePCD_dialog
			// 
			this->savePCD_dialog->DefaultExt = L"pcd";
			// 
			// MyForm
			// 
			
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(784, 449);
			this->Controls->Add(this->savePCD_button);
			this->Controls->Add(this->saveFrame_button);
			this->Controls->Add(this->stat_strip);
			this->Controls->Add(this->MainStreamTab);
			this->Controls->Add(this->dstream_button);
			this->Controls->Add(this->cstream_button);
			this->Controls->Add(this->device_status);
			this->Controls->Add(this->Ldevice_status);
			this->Controls->Add(this->device_button);
			this->Controls->Add(this->shapeContainer1);
			this->Name = L"MyForm";
			this->Text = L"DS311 STREAM";
			this->Load += gcnew System::EventHandler(this, &MyForm::MyForm_Load);
			this->MainStreamTab->ResumeLayout(false);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
		private : void dstreamOff(){
			dstream_button->Text = L"Depth Stream";
			d_status->FillColor = System::Drawing::Color::Maroon;
			d_status->FillGradientColor = System::Drawing::Color::Maroon;
			MyForm::isDStreamOn = false;		  
		}

		private : void cstreamOff(){
			cstream_button->Text = L"Color Stream";
			c_status->FillColor = System::Drawing::Color::Maroon;
			c_status->FillGradientColor = System::Drawing::Color::Maroon;
			MyForm::isCStreamOn = false;		  
		}

		//when device is disconnected, stops all operations via this
		private : void deviceOff(){
			MyForm::dstreamOff();
			MyForm::cstreamOff();
		}

		//drawing openCV mat onto windows form box. In this instance, the cstream box
		public : void DrawCVColorFrame(cv::Mat& colorImage){
					  System::Drawing::Graphics^ graphics = this->cstream_box->CreateGraphics();
					  System::IntPtr ptr(colorImage.ptr());
					  System::Drawing::Bitmap^ b = gcnew System::Drawing::Bitmap(colorImage.cols, colorImage.rows, colorImage.step, 
						  System::Drawing::Imaging::PixelFormat::Format24bppRgb,ptr);
					  System::Drawing::RectangleF rect(this->cstream_box->Location.X, this->cstream_box->Location.Y, 
						  this->cstream_box->Width, this->cstream_box->Height);
					  graphics->DrawImage(b, rect);
		}

		private: System::Void MyForm_Load(System::Object^  sender, System::EventArgs^  e) {
			//HOWABOUT: initiate stuffs here as the UI inflates
		}
				 
		private: System::Void device_buttonClick(System::Object^  sender, System::EventArgs^  e) {
					 //do you need to catch if unable to stop device or start device?
					 if(device_button->Text == "Start Device"){
						//TODO : initialize device retriever here
				
						//TODO: when acknowledgement from device, do this
						//changes device button to stop device
						device_button->Text = L"Stop Device";
						device_status->Text = L"Online";
						device_status->ForeColor = System::Drawing::Color::Green;
						MyForm::isDeviceConnected = true;

					 }else if(device_button->Text == "Stop Device"){
						//TODO : terminate device using device retriever here
				 
						//TODO: when acknowledgement from device, do this
						 device_button->Text = L"Start Device";
						device_status->Text = L"Offline";
						device_status->ForeColor = System::Drawing::Color::OrangeRed;
						MyForm::isDeviceConnected = false;
						MyForm::deviceOff();
					 }
				 }

		private: System::Void dstream_buttonClick(System::Object^  sender, System::EventArgs^  e) {
				//TODO : starts stream, waits for stream live acknowledgement

				//if acknowledged, button turns green.
				//HOWABOUT: acknowledge baru to the if
				if(MyForm::isDeviceConnected){
					if(!MyForm::isDStreamOn){
						dstream_button->Text = L"Stop Stream";
				  		this->d_status->FillColor = System::Drawing::Color::Green;
						this->d_status->FillGradientColor = System::Drawing::Color::Green;
						MyForm::isDStreamOn = true;
					}else{
						dstream_button->Text = L"Depth Stream";
						this->d_status->FillColor = System::Drawing::Color::Maroon;
						this->d_status->FillGradientColor = System::Drawing::Color::Maroon;
						MyForm::isDStreamOn = false;
					}
				}else{
					//pop up window here
					MessageBox::Show("Device is not Online! Activate the Device first.");
				}
			}

		private: System::Void cstream_buttonClick(System::Object^  sender, System::EventArgs^  e) {
			//TODO : starts stream, waits for stream live acknowledgement

			//if acknowledged, button turns green.
			//HACK : acknowledge baru to the if
			if(MyForm::isDeviceConnected){
				if(!MyForm::isCStreamOn){
					 cstream_button->Text = L"Stop Stream";
					 this->c_status->FillColor = System::Drawing::Color::Green;
					 this->c_status->FillGradientColor = System::Drawing::Color::Green;

					 MyForm::isCStreamOn = true;
				}else{
					cstream_button->Text = L"Color Stream";
					this->c_status->FillColor = System::Drawing::Color::Maroon;
					this->c_status->FillGradientColor = System::Drawing::Color::Maroon;
					MyForm::isCStreamOn = false;
				}
			}else{
				//pop up window here
				MessageBox::Show("Device is not Online! Activate the Device first.");
				
			}
		}

	private: System::Void saveframe_buttonClick(System::Object^  sender, System::EventArgs^  e) {
		if(MyForm::isDeviceConnected){
			//TODO: get jpg file here from opencv
			this->saveFrame_dialog->DefaultExt = "jpg";
			this->saveFrame_dialog->ShowDialog();
			this->saveFrame_dialog->Filter = "JPeg Image | *.jpg| All Files|*.*";
			if(this->saveFrame_dialog->ShowDialog() == System::Windows::Forms::DialogResult::OK){
				//save file using opencv io instead
				if(saveFrame_dialog->OpenFile() != nullptr){
					
				}

			}
			
		}else{
			//POP
			MessageBox::Show("Device is not Online! Activate the Device first.");
		}	 
	}

	private: System::Void savePCD_buttonClick(System::Object^  sender, System::EventArgs^  e) {
		if(MyForm::isDeviceConnected){
			//TODO: get pcd file here from whatever
			this->savePCD_dialog->DefaultExt = "pcd";
			this->savePCD_dialog->ShowDialog();
		}else{
			//POP
			MessageBox::Show("Device is not Online! Activate the Device first.");
		}	 
	}

	private: System::Void statusStrip1_ItemClicked(System::Object^  sender, System::Windows::Forms::ToolStripItemClickedEventArgs^  e) {
			 }
};
}
