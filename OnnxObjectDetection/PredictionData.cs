using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System.Drawing;

namespace OnnxObjectDetection
{
   class PredictionData
    {
      /// <summary>
      /// La bitmap
      /// </summary>
      [ColumnName("bitmap")]
      [ImageType(640, 640)]
      public Bitmap Image { get; set; }
      /// <summary>
      /// Larghezza immagine
      /// </summary>
      [ColumnName("width")]
      public float ImageWidth => Image.Width;
      /// <summary>
      /// Altezza immagine
      /// </summary>
      [ColumnName("height")]
      public float ImageHeight => Image.Height;
   }
}