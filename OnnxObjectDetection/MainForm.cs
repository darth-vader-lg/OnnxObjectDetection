using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
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
         DrawObjectOnBitmap(bmp, prediction.GetResults());


         pictureBox.Image = bmp;
      }
      /// <summary>
      /// Marca l'immagine
      /// </summary>
      /// <param name="bmp">Bitmap da marcare</param>
      /// <param name="results">Risultati</param>
      private static void DrawObjectOnBitmap(Bitmap bmp, IEnumerable<PredictionResult.Result> results)
      {
         var categories = new[] { "Carp" };
         using var graphic = Graphics.FromImage(bmp); graphic.SmoothingMode = SmoothingMode.AntiAlias;
         foreach (var result in results) {
            var rect = new Rectangle(
               (int)(result.Box.Left),
               (int)(result.Box.Top),
               (int)(result.Box.Width),
               (int)(result.Box.Height));
            using var pen = new Pen(Color.Lime, Math.Max(Math.Min(rect.Width, rect.Height) / 320f, 1f));
            graphic.DrawRectangle(pen, rect);
            var fontSize = Math.Min(bmp.Size.Width, bmp.Size.Height) / 40f;
            fontSize = Math.Max(fontSize, 8f);
            fontSize = Math.Min(fontSize, rect.Height);
            using var font = new Font("Verdana", fontSize, GraphicsUnit.Pixel);
            var p = new Point(rect.Left, rect.Top);
            var text = $"{categories[result.Category]}:{(int)(result.Confidence * 100)}";
            var size = graphic.MeasureString(text, font);
            using var brush = new SolidBrush(Color.FromArgb(50, Color.Lime));
            graphic.FillRectangle(brush, p.X, p.Y, size.Width, size.Height);
            graphic.DrawString(text, font, Brushes.Black, p);
         }
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
