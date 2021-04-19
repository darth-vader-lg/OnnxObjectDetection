
namespace OnnxObjectDetection
{
   partial class MainForm
   {
      /// <summary>
      ///  Required designer variable.
      /// </summary>
      private System.ComponentModel.IContainer components = null;

      /// <summary>
      ///  Clean up any resources being used.
      /// </summary>
      /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
      protected override void Dispose(bool disposing)
      {
         if (disposing && (components != null)) {
            components.Dispose();
         }
         base.Dispose(disposing);
      }

      #region Windows Form Designer generated code

      /// <summary>
      ///  Required method for Designer support - do not modify
      ///  the contents of this method with the code editor.
      /// </summary>
      private void InitializeComponent()
      {
         this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
         this.buttonLoad = new System.Windows.Forms.Button();
         this.pictureBox = new System.Windows.Forms.PictureBox();
         this.openFileDialog = new System.Windows.Forms.OpenFileDialog();
         this.tableLayoutPanel1.SuspendLayout();
         ((System.ComponentModel.ISupportInitialize)(this.pictureBox)).BeginInit();
         this.SuspendLayout();
         // 
         // tableLayoutPanel1
         // 
         this.tableLayoutPanel1.ColumnCount = 1;
         this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
         this.tableLayoutPanel1.Controls.Add(this.buttonLoad, 0, 1);
         this.tableLayoutPanel1.Controls.Add(this.pictureBox, 0, 0);
         this.tableLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
         this.tableLayoutPanel1.Location = new System.Drawing.Point(0, 0);
         this.tableLayoutPanel1.Name = "tableLayoutPanel1";
         this.tableLayoutPanel1.RowCount = 2;
         this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
         this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle());
         this.tableLayoutPanel1.Size = new System.Drawing.Size(800, 450);
         this.tableLayoutPanel1.TabIndex = 0;
         // 
         // buttonLoad
         // 
         this.buttonLoad.Location = new System.Drawing.Point(3, 424);
         this.buttonLoad.Name = "buttonLoad";
         this.buttonLoad.Size = new System.Drawing.Size(75, 23);
         this.buttonLoad.TabIndex = 0;
         this.buttonLoad.Text = "Load...";
         this.buttonLoad.UseVisualStyleBackColor = true;
         this.buttonLoad.Click += new System.EventHandler(this.buttonLoad_Click);
         // 
         // pictureBox
         // 
         this.pictureBox.Dock = System.Windows.Forms.DockStyle.Fill;
         this.pictureBox.Location = new System.Drawing.Point(3, 3);
         this.pictureBox.Name = "pictureBox";
         this.pictureBox.Size = new System.Drawing.Size(794, 415);
         this.pictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
         this.pictureBox.TabIndex = 1;
         this.pictureBox.TabStop = false;
         // 
         // openFileDialog
         // 
         this.openFileDialog.Filter = "Image files|*.jpg;*.jpeg;*.bmp;*.png|All files|*.*";
         // 
         // MainForm
         // 
         this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
         this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
         this.ClientSize = new System.Drawing.Size(800, 450);
         this.Controls.Add(this.tableLayoutPanel1);
         this.Name = "MainForm";
         this.Text = "ONNX object detection ";
         this.tableLayoutPanel1.ResumeLayout(false);
         this.tableLayoutPanel1.PerformLayout();
         ((System.ComponentModel.ISupportInitialize)(this.pictureBox)).EndInit();
         this.ResumeLayout(false);

      }

      #endregion

      private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
      private System.Windows.Forms.Button buttonLoad;
      private System.Windows.Forms.PictureBox pictureBox;
      private System.Windows.Forms.OpenFileDialog openFileDialog;
   }
}

