using Microsoft.ML.Data;
using System.Drawing;

namespace OnnxObjectDetection
{
   class PredictionData
   {
      #region Fields
      /// <summary>
      /// Size dell'immagine
      /// </summary>
      private Size imageSize;
      #endregion
      #region Properties
      /// <summary>
      /// Il path dell'immagine
      /// </summary>
      [ColumnName("ImagePath")]
      public string ImagePath { get; set; }
      /// <summary>
      /// Larghezza immagine
      /// </summary>
      [ColumnName("width")]
      public float ImageWidth => imageSize == default ? (imageSize = Image.FromFile(ImagePath).Size).Width : imageSize.Width;
      /// <summary>
      /// Altezza immagine
      /// </summary>
      [ColumnName("height")]
      public float ImageHeight => imageSize == default ? (imageSize = Image.FromFile(ImagePath).Size).Height : imageSize.Height;
      #endregion
   }
}