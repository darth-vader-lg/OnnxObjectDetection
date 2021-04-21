namespace OnnxObjectDetection
{
   /// <summary>
   /// Configurazione del modello
   /// </summary>
   class ModelConfiguration
   {
      #region Properties
      /// <summary>
      /// Larghezza dellímmagine del modello
      /// </summary>
      public int ImageWidth { get; set; } = 640;
      /// <summary>
      /// Altezza dellímmagine del modello
      /// </summary>
      public int ImageHeight { get; set; } = 640;
      /// <summary>
      /// Nome del tensore di input
      /// </summary>
      public string InputName { get; set; } = "images";
      /// <summary>
      /// Nomi dei tensori di input
      /// </summary>
      public string[] OutputNames { get; set; } = new[] { "output1", "output2", "output3" };
      #endregion
   }
}
