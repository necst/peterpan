#include <filesystem>

template <typename MovingImageType, typename OutputImageType, typename OutputImageType2D>
void RescaleAndWriteImages(
    typename MovingImageType::Pointer fixedImage, const std::string& outputDirectory, const std::string& imageName) {
    using RescaleFilterType = itk::RescaleIntensityImageFilter<MovingImageType, OutputImageType>;
    using ExtractFilterType = itk::ExtractImageFilter<OutputImageType, OutputImageType2D>;
    using WriterType = itk::ImageFileWriter<OutputImageType2D>;

    auto rescaleFilter = RescaleFilterType::New();
    rescaleFilter->SetInput(fixedImage);
    rescaleFilter->SetOutputMinimum(0);
    rescaleFilter->SetOutputMaximum(255);

    try {
        rescaleFilter->Update();
    } catch (itk::ExceptionObject& ex) {
        std::cerr << "Exception caught while rescaling intensities: " << ex << std::endl;
        throw;
    }

    auto rescaled_output = rescaleFilter->GetOutput();
    auto size = rescaled_output->GetLargestPossibleRegion().GetSize();

    for (unsigned int i = 0; i < size[2]; ++i) {
        typename OutputImageType::RegionType extractRegion = rescaled_output->GetLargestPossibleRegion();
        extractRegion.SetSize(2, 0);
        extractRegion.SetIndex(2, i);

        auto extractFilter = ExtractFilterType::New();
        extractFilter->SetExtractionRegion(extractRegion);
        extractFilter->SetInput(rescaled_output);
        extractFilter->SetDirectionCollapseToIdentity();

        auto writer = WriterType::New();
        std::ostringstream outputFilePath;
        outputFilePath << outputDirectory << "/" << imageName << std::setw(4) << std::setfill('0') << i << ".png";
        writer->SetFileName(outputFilePath.str());
        writer->SetInput(extractFilter->GetOutput());

        try {
            writer->Update();
            std::cout << imageName << " slice written to: " << outputFilePath.str() << std::endl;
        } catch (itk::ExceptionObject& error) {
            std::cerr << "Error writing the fixed image slice: " << error << std::endl;
            throw;
        }
    }
}

namespace fs = std::filesystem;

template <typename ImageType>
void ReadImageSeriesPNG(const std::string& directory, typename ImageType::Pointer& image) {
    using ImageIOType = itk::PNGImageIO;
    using ImageReaderType = itk::ImageSeriesReader<ImageType>;

    // List all PNG files in the directory
    std::vector<std::string> imageNames;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".png") {
            imageNames.push_back(entry.path().string());
        }
    }

    if (imageNames.empty()) {
        std::cerr << "Error: No PNG files found in directory: " << directory << std::endl;
        throw std::runtime_error("No PNG files found");
    }

    std::sort(imageNames.begin(), imageNames.end());

    std::cout << "Number of images: " << imageNames.size() << std::endl;
    std::cout << "First image file: " << imageNames[0] << std::endl;

    for (const auto& imageName : imageNames) {
        std::cout << "File: " << imageName << std::endl;
    }

    auto imageIO = ImageIOType::New();
    auto imageReader = ImageReaderType::New();
    imageReader->SetImageIO(imageIO);
    imageReader->SetFileNames(imageNames);
    imageReader->Update();

    image = imageReader->GetOutput();

    std::cout << "Image dimensions: " << image->GetLargestPossibleRegion().GetSize() << std::endl;
    std::cout << "Image spacing: " << image->GetSpacing() << std::endl;
    std::cout << "Image origin: " << image->GetOrigin() << std::endl;
    std::cout << "Image direction: " << image->GetDirection() << std::endl;
}

template <typename MovingImageType, typename FixedImageType, typename TransformType>
typename MovingImageType::Pointer ResampleImage(
    typename MovingImageType::Pointer movingImage,
    typename FixedImageType::Pointer fixedImage,
    typename TransformType::Pointer finalTransform) {
    using ResampleFilterType = itk::ResampleImageFilter<MovingImageType, FixedImageType>;

    auto resampleFilter = ResampleFilterType::New();
    resampleFilter->SetTransform(finalTransform);
    resampleFilter->SetInput(movingImage);
    resampleFilter->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
    resampleFilter->SetOutputSpacing(fixedImage->GetSpacing());
    resampleFilter->SetOutputOrigin(fixedImage->GetOrigin());
    resampleFilter->SetOutputDirection(fixedImage->GetDirection());
    resampleFilter->SetInterpolator(itk::LinearInterpolateImageFunction<MovingImageType, double>::New());
    resampleFilter->Update();
    std::cout << "Resempled Image Size: " << resampleFilter->GetOutput()->GetLargestPossibleRegion().GetSize()
              << std::endl;
    std::cout << "Resampling done" << std::endl;
    // Return the resampled image
    return resampleFilter->GetOutput();
}

template <typename ImageType>
void ReadImageSeries(const std::string& directory, typename ImageType::Pointer& image) {
    using ImageIOType = itk::GDCMImageIO;
    using NamesGeneratorType = itk::GDCMSeriesFileNames;
    using ImageReaderType = itk::ImageSeriesReader<ImageType>;

    auto imageIO = ImageIOType::New();
    auto namesGenerator = NamesGeneratorType::New();
    namesGenerator->SetDirectory(directory);

    std::vector<std::string> imageNames = namesGenerator->GetInputFileNames();

    if (imageNames.empty()) {
        std::cerr << "Error: No DICOM files found in directory: " << directory << std::endl;
        throw std::runtime_error("No DICOM files found");
    }

    std::sort(imageNames.begin(), imageNames.end());

    std::cout << "Number of images: " << imageNames.size() << std::endl;
    std::cout << "First image file: " << imageNames[0] << std::endl;

    for (const auto& imageName : imageNames) {
        std::cout << "File: " << imageName << std::endl;
    }

    auto imageReader = ImageReaderType::New();
    imageReader->SetImageIO(imageIO);
    imageReader->SetFileNames(imageNames);
    imageReader->Update();

    image = imageReader->GetOutput();

    std::cout << "Image dimensions: " << image->GetLargestPossibleRegion().GetSize() << std::endl;
    std::cout << "Image spacing: " << image->GetSpacing() << std::endl;
    std::cout << "Image origin: " << image->GetOrigin() << std::endl;
    std::cout << "Image direction: " << image->GetDirection() << std::endl;
}

// Function to compute the geometric center of the volume
template <typename ImageType>
typename ImageType::PointType ComputeImageCenter(const typename ImageType::Pointer& image) {
    typename ImageType::PointType center;
    typename ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();

    for (unsigned int i = 0; i < ImageType::ImageDimension; ++i) {
        std::cout << "Dimension: " << i << std::endl;
        std::cout << "Size: " << size[i] << std::endl;
        std::cout << "Spacing: " << image->GetSpacing()[i] << std::endl;
        std::cout << "Origin: " << image->GetOrigin()[i] << std::endl;
        center[i] = 0.5 * (size[i]) * image->GetSpacing()[i] + image->GetOrigin()[i];
    }

    return center;
}