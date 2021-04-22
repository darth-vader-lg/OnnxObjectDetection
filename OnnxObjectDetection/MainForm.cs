using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
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
         // Apertura dialog di caricamento file
         if (openFileDialog.ShowDialog(this) != DialogResult.OK)
            return;
         // Previsione
         var prediction = model.Predictor.Predict(new PredictionData { ImagePath = openFileDialog.FileName });
         // Disegna i riquadri e il punteggio sullímmagine
         var bmp = new Bitmap(Image.FromFile(openFileDialog.FileName));
         DrawObjectOnBitmap(bmp, prediction.GetResults());
         // Visualizza l'immagine
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
         using var graphic = Graphics.FromImage(bmp);
         foreach (var result in results) {
            var rect = new Rectangle(
               (int)(result.Box.Left),
               (int)(result.Box.Top),
               (int)(result.Box.Width),
               (int)(result.Box.Height));
            using var pen = new Pen(Color.Lime, Math.Max(Math.Min(rect.Width, rect.Height) / 320f, 1f));
            graphic.SmoothingMode = SmoothingMode.AntiAlias;
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
            graphic.SmoothingMode = SmoothingMode.None;
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
         // Crea il task di aggiornamento
         var msgForm = new Form
         {
            FormBorderStyle = FormBorderStyle.None,
            StartPosition = FormStartPosition.CenterScreen,
            Size = new Size(200, 50)
         };
         msgForm.Controls.Add(new Label
         {
            Text = "Preparing the model...",
            Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.MiddleCenter,
         });
         new Task(async () =>
         {
            while (!msgForm.Modal)
               await Task.Delay(10);
            var onnxModelName = "carp.onnx";
            await Task.Run(() =>
            {
               var onnxPath = Path.Combine("..", "..", "..", onnxModelName);
               var zipPath = Path.GetFileNameWithoutExtension(onnxPath) + ".model.zip";
               if (!File.Exists(zipPath) || File.GetLastWriteTime(onnxPath) > File.GetLastWriteTime(zipPath)) {
                  model = new(ml, onnxPath);
                  ml.Model.Save(model, ml.Data.LoadFromEnumerable(Array.Empty<PredictionData>()).Schema, zipPath);
               }
               else
                  model = new(ml, zipPath);
            });
            msgForm.DialogResult = DialogResult.OK;
         }).RunSynchronously();
         msgForm.ShowDialog(this);
      }
      #endregion
   }
}
