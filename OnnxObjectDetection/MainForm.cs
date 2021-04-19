using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace OnnxObjectDetection
{
   /// <summary>
   /// Form principale
   /// </summary>
   public partial class MainForm : Form
   {
      #region Fields
      /// <summary>
      /// Modello
      /// </summary>
      private Model model;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public MainForm()
      {
         InitializeComponent();
      }
      /// <summary>
      /// Click su pulsante di caricamento immagine
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void buttonLoad_Click(object sender, EventArgs e)
      {
         if (openFileDialog.ShowDialog(this) != DialogResult.OK)
            return;
         var bmp = new Bitmap(Image.FromFile(openFileDialog.FileName));
         var prediction = model.Predictor.Predict(new PredictionData { Image = bmp });
         var r = prediction.GetResults(new[] { "Carp" }, bmp);
      }
      /// <summary>
      /// Funzione di caricamento del form
      /// </summary>
      /// <param name="e"></param>
      protected override void OnLoad(EventArgs e)
      {
         base.OnLoad(e);
         var ml = new MLContext();
         model = new(
            ml,
            Path.Combine("..", "..", "..", "carp.onnx"),
            new[] { "Carp" },
            640,
            640);
      }
      #endregion
   }
}
