#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/ext/vector_float3.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <glm/ext/matrix_float4x4.hpp>
#include <limits>
#include <ostream>
#include <set>
#include <optional>
#include <stdexcept>
#include <string>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>
#include <GLFW/glfw3.h>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cstdio>
#include <string_view>

#include <ft2build.h>
#include FT_FREETYPE_H

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define DEBUG_LEVEL 3

constexpr uint32_t WIDTH = 1400;
constexpr uint32_t HEIGHT = 1000;

const std::string MODEL_PATH = "../models/viking_room.obj";
const std::string TEXTURE_PATH = "../textures/viking_room.png";
const std::string FONT_PATH = "../fonts/RibeyeMarrow-Regular.ttf";

constexpr uint32_t FONT_ATLAS_WIDTH = 1024;
constexpr uint32_t FONT_ATLAS_HEIGHT = 512;
constexpr uint32_t FONT_PIXEL_SIZE = 35;
constexpr size_t MAX_UI_VERTICES = 4096;

const int MAX_FRAMES_IN_FLIGHT = 2;

uint32_t currentFrame = 0;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// swapchain扩展:用于准备 Image, 实现双重缓冲与三重缓冲,获取新画布以及present
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec4 color;
    glm::vec2 texCoord;
    glm::vec3 barycentric;

    // 获取处理顶点时的 步长
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;//绑定号
        bindingDescription.stride = sizeof(Vertex);//计算前面pos+color的大小作为对齐步长: 28字节
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;//每处理一个顶点移动一个步长
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        attributeDescriptions[3].binding = 0;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[3].offset = offsetof(Vertex, barycentric);
        return attributeDescriptions;
    }
};

enum DrawMode : uint32_t {
    DRAW_MODEL_FILL = 0,
    DRAW_MODEL_WIRE = 1,
    DRAW_UI = 2
};

struct DrawPushConstants {
    float viewportWidth = 1.0f;
    float viewportHeight = 1.0f;
    float lineWidthPx = 0.4f;
    float featherPx = 1.0f;
    uint32_t mode = DRAW_MODEL_FILL;
};

static_assert(sizeof(DrawPushConstants) == 20);

struct Glyph {
    glm::ivec2 size{0};
    glm::ivec2 bearing{0};
    float advance = 0.0f;
    glm::vec2 uv0{0.0f};
    glm::vec2 uv1{0.0f};
    bool valid = false;
};

struct UiRect {
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;

    bool contains(const glm::vec2& point) const {
        return point.x >= x && point.y >= y &&
               point.x < x + width && point.y < y + height;
    }
};


struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct AABB {
    glm::vec3 lo = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3 hi = glm::vec3(std::numeric_limits<float>::lowest());

    void expand(const glm::vec3& p) {
        lo = glm::min(lo, p);
        hi = glm::max(hi, p);
    }
    
    void expand(const AABB& other) {
        expand(other.lo);
        expand(other.hi);
    }
};

struct BVHPrimitive {
    uint32_t triangle = 0;
    AABB bounds;
    glm::vec3 centroid{0.0f};
};

struct BVHNode {
    AABB bounds;
    uint32_t left = UINT32_MAX;
    uint32_t right = UINT32_MAX;
    uint32_t first = 0;
    uint32_t count = 0;
    bool isLeaf() const {
        return count > 0;
    }
};

struct PickHit {
    bool hit = false;
    float distance = std::numeric_limits<float>::max();
    uint32_t triangle = UINT32_MAX;
    glm::vec3 worldPosition{0.0f};
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

class VulkanHut {
public:

    void run() {
        initWindow();//用GLFW创建一个窗口
        initVulkan();
        mainLoop();
        cleanup();
    }

private:

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    GLFWwindow* window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkSurfaceKHR surface;
    VkQueue presentQueue;
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline fillPipeline = VK_NULL_HANDLE;
    VkPipeline wirePipeline = VK_NULL_HANDLE;
    VkPipeline uiPipeline = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkDescriptorSetLayout descriptorSetLayout;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    bool framebufferResized = false;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;

    uint32_t mipLevels;
    VkImageView textureImageView;
    VkSampler textureSampler;

    VkImage fontImage = VK_NULL_HANDLE;
    VkDeviceMemory fontImageMemory = VK_NULL_HANDLE;
    VkImageView fontImageView = VK_NULL_HANDLE;
    VkSampler fontSampler = VK_NULL_HANDLE;
    std::array<Glyph, 128> glyphs{};
    glm::vec2 solidUv{
        0.5f / static_cast<float>(FONT_ATLAS_WIDTH),
        0.5f / static_cast<float>(FONT_ATLAS_HEIGHT)
    };

    std::vector<VkBuffer> uiVertexBuffers;
    std::vector<VkDeviceMemory> uiVertexBufferMemories;
    std::vector<void*> uiVertexBufferMapped;
    std::array<uint32_t, MAX_FRAMES_IN_FLIGHT> uiVertexCounts{};
    std::vector<Vertex> uiVertices;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;

    std::vector<BVHPrimitive> bvhPrimitives;
    std::vector<BVHNode> bvhNodes;

    glm::vec3 modelPivot{0.0f};
    float modelRadius = 1.0f;
    glm::quat modelRotation{1.0f, 0.0f, 0.0f, 0.0f};

    glm::vec3 cameraDirection = glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f));

    float cameraDistance = 3.464f;
    float minCameraDistance = 0.1f;
    float maxCameraDistance = 100.0f;

    enum class DragTarget {
        None,
        Model,
        RotationSpeed
    };

    DragTarget dragTarget = DragTarget::None;
    glm::vec3 dragStartBall{0.0f};
    glm::quat dragStartRotation{1.0f, 0.0f, 0.0f, 0.0f};

    bool wireframe = false;
    float displayedFps = 0.0f;
    float rotationSpeedDeg = 20.0f;
    float automaticRotationRadians = 0.0f;
    DrawPushConstants drawPush{};

    static constexpr float MIN_ROTATION_SPEED = 0.0f;
    static constexpr float MAX_ROTATION_SPEED = 180.0f;

    UiRect modeButton{20.0f, 160.0f, 208.0f, 36.0f};
    UiRect speedSliderHitArea{20.0f, 94.0f, 208.0f, 46.0f};
    UiRect speedSliderTrack{20.0f, 122.0f, 190.0f, 6.0f};

    uint32_t selectedTriangle = UINT32_MAX;

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();

        createSwapChain();
        createImageViews();
        createRenderPass();

        createDescriptorSetLayout();
        createGraphicsPipelines();
        createColorResources();
        createDepthResources();
        createFramebuffers();

        createCommandPool();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        createFontAtlas();
        createFontSampler();
        loadModel();

        buildBVH();

        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createUiVertexBuffers();
        createDescriptorPool();
        createDescriptorSets();

        createCommandBuffer();
        createSyncObjects();
    }

    void initWindow() {
        glfwInit(); //加载ddl, 初始化计时器和输入设备
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // 声明不创建OpenGL上下文

        glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE); // 指定窗口背景为透明

        window = glfwCreateWindow(WIDTH, HEIGHT, "vulkan", nullptr, nullptr);// 创建窗口获得窗口句柄
        glfwSetWindowUserPointer(window, this);// 把 this指针塞进 window 对象
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);// 注册:窗口大小改变回调函数

        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, cursorPositionCallback);
        glfwSetScrollCallback(window, scrollCallback    );
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<VulkanHut*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void mainLoop() {
        double lastFrameTime = glfwGetTime();
        double fpsStartTime = lastFrameTime;
        const char* COLOR_INFO    = "\033[36m";
        const char* COLOR_RESET   = "\033[0m";
        int frameCount = 0;
        while(!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            double currentTime = glfwGetTime();
            float deltaSeconds = static_cast<float>(currentTime - lastFrameTime);
            lastFrameTime = currentTime;
            deltaSeconds = std::min(deltaSeconds, 0.1f);

            if(dragTarget != DragTarget::Model) {
                automaticRotationRadians += glm::radians(rotationSpeedDeg) * deltaSeconds;
                automaticRotationRadians = std::fmod(automaticRotationRadians, 6.28318530718f);
            }

            drawFrame();
            frameCount++;
            if (currentTime - fpsStartTime >= 0.5) {
                displayedFps = static_cast<float>(frameCount / (currentTime - fpsStartTime));
                
                std::cout << COLOR_INFO << "\r[Vulkan Hut] FPS: " << static_cast<int>(displayedFps) << "    " << COLOR_RESET << std::flush;

                fpsStartTime = currentTime;
                frameCount = 0;
            }
        }
        vkDeviceWaitIdle(device);
    }

    void createInstance() {

        VkApplicationInfo appInfo = {};// 通知显卡信息便于特定优化(可以不写)
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "vulkan cube";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();// 获取 instance 的 extension (swapchain是device的extension)
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

        if(enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        if(enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        }
        else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        if(vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
        for(const char* layerName : validationLayers) {
            bool layerFound = false;
            for(const auto& layerProperties : availableLayers) {
                if(0 == strcmp(layerName, layerProperties.layerName)) {
                    layerFound = true;
                }
            }
            if(!layerFound) return false;
        }
        return true;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if(enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData) {
        
        const char* COLOR_RESET   = "\033[0m";
        const char* COLOR_VERBOSE = "\033[0m";
        const char* COLOR_INFO    = "\033[36m";
        const char* COLOR_WARNING = "\033[33m";
        const char* COLOR_ERROR   = "\033[31m";
        const char* color = COLOR_RESET;
        
        if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
            color = COLOR_ERROR;
        } else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            color = COLOR_WARNING;
        } else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
            color = COLOR_INFO;
        } else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
            color = COLOR_VERBOSE;
        }

        VkDebugUtilsMessageSeverityFlagBitsEXT minSeverity;
        if (DEBUG_LEVEL == 1) minSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
        else if (DEBUG_LEVEL == 2) minSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
        else if (DEBUG_LEVEL == 3) minSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
        else minSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

        if(messageSeverity >= minSeverity) {
            std::cerr << color << pCallbackData->pMessage << COLOR_RESET << std::endl;
        }

        return VK_FALSE;
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if(nullptr != func) {
            return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
        }
        else {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }

    void SubmitDebugUtilsMessageEXT(VkInstance instance, VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData) {
        auto func = (PFN_vkSubmitDebugUtilsMessageEXT) vkGetInstanceProcAddr(instance, "vkSubmitDebugUtilsMessageEXT");
        if(nullptr != func) {
            func(instance, messageSeverity, messageType, pCallbackData);
        }
    }

    void Log(const char* message, VkDebugUtilsMessageSeverityFlagBitsEXT severity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        if(!enableValidationLayers) return;

        const char* COLOR_INFO    = "\033[36m";
        const char* COLOR_RESET   = "\033[0m";
        std::string prefix = "[Debug Log]: ";
        std::string fullMessage = COLOR_INFO + prefix + message + COLOR_RESET;

        VkDebugUtilsMessengerCallbackDataEXT callbackData{};
        callbackData.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT;
        callbackData.pMessageIdName = "LOG";
        callbackData.messageIdNumber = 0;
        callbackData.pMessage = fullMessage.c_str();

        SubmitDebugUtilsMessageEXT(instance, severity, VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, &callbackData);
    }

    void setupDebugMessenger() {
        if(!enableValidationLayers) return;
        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        populateDebugMessengerCreateInfo(createInfo);
        if(CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger");
        }
    }

    void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (nullptr != func) {
            func(instance, debugMessenger, pAllocator);
        }
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {

        QueueFamilyIndices indices = findQueueFamilies(device);

        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for(const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        bool swapChainAdequate = false;
        if(requiredExtensions.empty()) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && requiredExtensions.empty() &&swapChainAdequate;
    }

    void pickPhysicalDevice() {
        // 首先获取所有显卡的handle
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if(0 == deviceCount) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for(const auto& device : devices) {
            if(isDeviceSuitable(device)) {
                physicalDevice = device;
                msaaSamples = getMaxUsableSampleCount();
                break;
            }
        }
        if(VK_NULL_HANDLE == physicalDevice) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

   
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        VkBool32 presentSupport = false;
        int i = 0;
        for(const auto& queueFamily : queueFamilies) {

            if(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if(presentSupport) {
                indices.presentFamily = i;
            }

            if(indices.isComplete()) break;
            i++;
        }
        return indices;
    }

    void createLogicalDevice() { // 创建逻辑设备 vkDevice

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1.0f;

        for(uint32_t queueFamily : uniqueQueueFamilies) {//寻找图形队列和显示队列
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);// 从逻辑设备中获取图形队列
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);// 获取显示队列
    }

    void createSurface() {
        if(VK_SUCCESS != glfwCreateWindowSurface(instance, window, nullptr, &surface)) {
            throw std::runtime_error("failed to create window surface");
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {

        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
        if(0 != formatCount) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        if(0 != presentModeCount) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for(const auto& availableFormat : availableFormats) {
            if(VK_FORMAT_B8G8R8A8_SRGB == availableFormat.format && VK_COLOR_SPACE_SRGB_NONLINEAR_KHR == availableFormat.colorSpace) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for(const auto& availablePresentMode : availablePresentModes) {
            if(VK_PRESENT_MODE_MAILBOX_KHR == availablePresentMode) {
                return availablePresentMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if(capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    void createSwapChain() {
        // 首先查询当前物理设备对这个窗口表面 swapchain 的支持情况 : 
        // capabilities (extent) : 最大最小 Image 数量,当前窗口尺寸,支持的 transform
        // formats : 图像格式和颜色空间
        // presentModes : 支持的显示模式(垂直同步) : FIFO, MAILBOX, IMMEDIATE
        // 最后会创建一个 swapChainImages (VkImage的集合)
        // vkAckquireNextImageKHR() 获取下一张可渲染图像
        // vkQueuePresentKHR() 把图像提交到屏幕
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = choosePresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        std::cout <<"extent:"<<"height:"<< extent.height << " width:" << extent.width << std::endl;

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if(swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainKHR oldSwapchain = swapChain;

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface; // 将 swapchain 绑定 surface
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;// 像素格式
        createInfo.imageColorSpace = surfaceFormat.colorSpace; // 色彩空间
        createInfo.imageExtent = extent;// 宽和高
        createInfo.imageArrayLayers = 1;// swapchain image的数组层数
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;// 这些image会被用作颜色附件(直接把渲染结果画在这些image上)

        // 处理present queue 和 graphics queue不一样的情况
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        if(indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform; // 手机可能要横屏旋转

        // 查找支持的透明混合模式
        VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        if (swapChainSupport.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) {
            compositeAlpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
        } else if (swapChainSupport.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) {
            compositeAlpha = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
        } else if (swapChainSupport.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) {
            compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        }

        // 应用混合模式
        createInfo.compositeAlpha = compositeAlpha;

        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        createInfo.oldSwapchain = oldSwapchain;

        if(VK_SUCCESS != vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain)) {
            throw std::runtime_error("failed to create swap chain!");
        }

        if(VK_NULL_HANDLE != oldSwapchain) {
            vkDestroySwapchainKHR(device, oldSwapchain, nullptr);
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

    }

    void createImageViews() { // 为 SwapchainImages 创建每一个 Image
        swapChainImageViews.resize(swapChainImages.size());
        for(size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; // 指定是 2D 图像
            createInfo.format = swapChainImageFormat; // 上个函数保存的色彩格式
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;// 此处区分颜色图像/深度图像/模板图像
            createInfo.subresourceRange.baseMipLevel = 0;// 指定从第0层 mipmap 开始
            createInfo.subresourceRange.levelCount = 1;// 这个imageView 只覆盖一个 mip level
            createInfo.subresourceRange.baseArrayLayer = 0;// 指定从数组层第0层开始
            createInfo.subresourceRange.layerCount = 1;// 指定 view 覆盖1个array layer

            if(VK_SUCCESS != vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i])) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    void createGraphicsPipelines() {
        auto vertShaderCode = readFile("../shaders/shader.vert.spv");
        auto fragShaderCode = readFile("../shaders/shader.frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertStage{};
        vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertStage.module = vertShaderModule;
        vertStage.pName = "main";

        VkPipelineShaderStageCreateInfo fragStage{};
        fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragStage.module = fragShaderModule;
        fragStage.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertStage, fragStage};

        std::array<VkDynamicState, 2> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &bindingDescription;
        vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInput.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = msaaSamples;
        multisampling.sampleShadingEnable = VK_FALSE;

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(DrawPushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;

        if(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout");
        }

        auto createPipeline = [&](bool depthTest,
                                  bool depthWrite,
                                  VkCompareOp depthCompare,
                                  bool blend,
                                  VkColorComponentFlags colorWriteMask,
                                  VkCullModeFlags cullMode,
                                  VkPipeline& output) {
            VkPipelineRasterizationStateCreateInfo rasterizer{};
            rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.depthClampEnable = VK_FALSE;
            rasterizer.rasterizerDiscardEnable = VK_FALSE;
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.cullMode = cullMode;
            rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rasterizer.depthBiasEnable = VK_FALSE;
            rasterizer.lineWidth = 1.0f;

            VkPipelineDepthStencilStateCreateInfo depthStencil{};
            depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencil.depthTestEnable = depthTest ? VK_TRUE : VK_FALSE;
            depthStencil.depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE;
            depthStencil.depthCompareOp = depthCompare;
            depthStencil.depthBoundsTestEnable = VK_FALSE;
            depthStencil.stencilTestEnable = VK_FALSE;

            VkPipelineColorBlendAttachmentState colorBlendAttachment{};
            colorBlendAttachment.colorWriteMask = colorWriteMask;
            colorBlendAttachment.blendEnable = blend ? VK_TRUE : VK_FALSE;
            colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
            colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

            VkPipelineColorBlendStateCreateInfo colorBlending{};
            colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlending.logicOpEnable = VK_FALSE;
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;

            VkGraphicsPipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = shaderStages;
            pipelineInfo.pVertexInputState = &vertexInput;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = &depthStencil;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = &dynamicState;
            pipelineInfo.layout = pipelineLayout;
            pipelineInfo.renderPass = renderPass;
            pipelineInfo.subpass = 0;

            if(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &output) != VK_SUCCESS) {
                throw std::runtime_error("failed to create graphics pipeline");
            }
        };

        constexpr VkColorComponentFlags RGBA =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;

        createPipeline(true, true, VK_COMPARE_OP_LESS, false, RGBA,
                       VK_CULL_MODE_BACK_BIT, fillPipeline);
        // X-Ray 线框：不测试/写入深度，也不剔除背面，所有三角形边都可见。
        createPipeline(false, false, VK_COMPARE_OP_ALWAYS, true, RGBA,
                       VK_CULL_MODE_NONE, wirePipeline);
        createPipeline(false, false, VK_COMPARE_OP_ALWAYS, true, RGBA,
                       VK_CULL_MODE_NONE, uiPipeline);

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if(!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }
        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        return buffer;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule shaderModule;
        if(VK_SUCCESS != vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule)) {
            throw std::runtime_error("failed to create shader module!");
        }
        return shaderModule;
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};// 这个 attachment 对应render pass里用到的图像目标 (此时是swapchain Image)
        colorAttachment.format = swapChainImageFormat;// 复用之前保存的Image的format
        colorAttachment.samples = msaaSamples;//采样数
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;// render pass开始时如何处理这张 attachment的原有内容
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;// render pass 后如何处理渲染的结果
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;// render pass开始前, 这个attachment期望的布局
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;// render pass结束后, 这个Image要变成present engine可以使用的布局
        
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = msaaSamples;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription colorAttachmentResolve{};
        colorAttachmentResolve.format = swapChainImageFormat;
        colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;// 第0个引用
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;//在subpass期间使用颜色附件的最佳布局

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentResolveRef{};
        colorAttachmentResolveRef.attachment = 2;
        colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};// 一个render pass 可以有多个subpass, 此时只有颜色渲染
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;// 分为图形管线和计算管线
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;
        subpass.pResolveAttachments = &colorAttachmentResolveRef;


        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 3> attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if(VK_SUCCESS != vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass)) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createFramebuffers() {
        // 绑定 render pass 的attachment和 image view
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for(size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<VkImageView, 3> attachments = {
                colorImageView,
                depthImageView,
                swapChainImageViews[i]
            };
            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;// 这里的尺寸 framebuffer要覆盖整张swapchain image
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;// 2D 层数为1

            if(VK_SUCCESS != vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i])) {
                throw std::runtime_error("failed to create framebuffer");
            }
        }
    }

    void createCommandPool() {// 创建 command pool必须绑定 queue family
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;// 允许单独reset从这个pool分配出的command buffer
        // 用于每一帧重新录制 command buffer
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();// 指定刚刚找到的graphics queue
        if(VK_SUCCESS != vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool)) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        // 创建 staging buffer : 
        // HOST_VISIBLE : CPU可以直接访问
        // HOST_COHERENT : CPU 写入后, 不需要手动 flush, GPU 也能看到更新
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data); // map 即将显存映射到CPU的地址空间
        memcpy(data, vertices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // 创建真正的 vertex buffer
        // TRANSFER_DST_BIT : 作为 copy 的目标 : 从 staging 到 vertex
        // VERTEX_BUFFER_BIT : 声明要绑定 vertex buffer
        // DEVICE_LOCAL_BIT : 这块内存位于本地显存
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

    }

    void createIndexBuffer() { 
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
    
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        // 临时分配一个 command buffer, 录制 copybuffer 的命令, 从而把 srcBuffer 中的内容拷贝 到 dstBuffer
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommand(commandBuffer);
    }


    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type!");
    }


    void createCommandBuffer() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();
        if(VK_SUCCESS != vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data())) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        // 每个 frame-in-flight 都有自己的 command buffer
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if(VK_SUCCESS != vkBeginCommandBuffer(commandBuffer, &beginInfo)) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }
        VkRenderPassBeginInfo renderPassInfo{};
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.003f, 0.003f, 0.003f, 0.7f}};
        clearValues[1].depthStencil = {1.0f, 0};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];

        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        drawPush.viewportWidth = static_cast<float>(swapChainExtent.width);
        drawPush.viewportHeight = static_cast<float>(swapChainExtent.height);

        auto pushMode = [&](uint32_t mode) {
            drawPush.mode = mode;
            vkCmdPushConstants(
                commandBuffer,
                pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(DrawPushConstants),
                &drawPush
            );
        };

        VkBuffer modelVertexBuffers[] = {vertexBuffer};
        VkDeviceSize modelOffsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, modelVertexBuffers, modelOffsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        if(wireframe) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wirePipeline);
            pushMode(DRAW_MODEL_WIRE);
            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        } else {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, fillPipeline);
            pushMode(DRAW_MODEL_FILL);
            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        }

        if(uiVertexCounts[currentFrame] > 0) {
            VkBuffer uiBuffers[] = {uiVertexBuffers[currentFrame]};
            VkDeviceSize uiOffsets[] = {0};

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, uiPipeline);
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, uiBuffers, uiOffsets);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
            pushMode(DRAW_UI);
            vkCmdDraw(commandBuffer, uiVertexCounts[currentFrame], 1, 0, 0);
        }

        vkCmdEndRenderPass(commandBuffer);
        if(VK_SUCCESS != vkEndCommandBuffer(commandBuffer)) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void createSyncObjects() {

        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(swapChainImages.size());
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if(VK_SUCCESS != vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) || VK_SUCCESS != vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i])) {
                throw std::runtime_error("failed to create semaphore!");
            }
        }
        for(size_t i = 0; i < swapChainImages.size(); i++) {
            if(VK_SUCCESS != vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i])) {
                throw std::runtime_error("failed to create semaphore!");
            }
        }
    }

    void drawFrame() {
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        uint32_t imageIndex;

        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if(VK_ERROR_OUT_OF_DATE_KHR == result) {
            recreateSwapChain();
            return;
        }
        else if(VK_SUCCESS != result && VK_SUBOPTIMAL_KHR != result) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        updateUniformBuffer(currentFrame);
        updateUiBuffer(currentFrame);

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[imageIndex]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if(VK_SUCCESS != vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame])) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if(VK_ERROR_OUT_OF_DATE_KHR == result || VK_SUBOPTIMAL_KHR == result || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if(VK_SUCCESS != result) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void cleanupSwapChain() {
        for(auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        vkDestroyImageView(device, colorImageView, nullptr);
        vkDestroyImage(device, colorImage, nullptr);
        vkFreeMemory(device, colorImageMemory, nullptr);

        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        for(auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
        swapChain = VK_NULL_HANDLE;
    }

    void destroyGraphicsPipelines() {
        vkDestroyPipeline(device, fillPipeline, nullptr);
        vkDestroyPipeline(device, wirePipeline, nullptr);
        vkDestroyPipeline(device, uiPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        fillPipeline = VK_NULL_HANDLE;
        wirePipeline = VK_NULL_HANDLE;
        uiPipeline = VK_NULL_HANDLE;
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while(0 == width || 0 == height) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device);
        cleanupSwapChain();
        destroyGraphicsPipelines();
        vkDestroyRenderPass(device, renderPass, nullptr);

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipelines();
        createColorResources();
        createDepthResources();
        createFramebuffers();
    }

    void createDescriptorSetLayout() {
        // 和 uniform buffer 有关, shader 访问GPU资源时用的资源信息描述记录, 分为
        // - uniform buffer descriptor
        // - image sampler descriptor
        // 创建 graphics pipeline 时, 需要知道 shader 会使用哪些 descriptor set, 每个set有哪些binding
        // uniform buffer 常用于 : MVP 矩阵, 光照参数, 材质常量, 少量全局参数
        // descriptor 存在的意义是为了让运行在 GPU 中的shader通过set和binding找到对应的资源
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;// 指定了 binding=0
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;// 注 : binding也可以是数组, 这里就会大于1
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;// 这个descriptor只能被vertex shader访问

        VkDescriptorSetLayoutBinding textureSamplerBinding{};
        textureSamplerBinding.binding = 1;
        textureSamplerBinding.descriptorCount = 1;
        textureSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        textureSamplerBinding.pImmutableSamplers = nullptr;
        textureSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding fontSamplerBinding{};
        fontSamplerBinding.binding = 2;
        fontSamplerBinding.descriptorCount = 1;
        fontSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        fontSamplerBinding.pImmutableSamplers = nullptr;
        fontSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
            uboLayoutBinding,
            textureSamplerBinding,
            fontSamplerBinding
        };
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);//首先计算每个UBO buffer的大小

        // 为每个帧准备一套 uniform buffer
        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }
    }
    // 为每一帧重新计算一组MVP矩阵, 并把它写入当前帧对应的 uniform buffer
    void updateUniformBuffer(uint32_t currentImage) {
        UniformBufferObject ubo{};
        ubo.model = getModelMatrix();
        ubo.view = getViewMatrix();
        ubo.proj = getProjectionMatrix();

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void createUiVertexBuffers() {
        VkDeviceSize bufferSize = MAX_UI_VERTICES * sizeof(Vertex);

        uiVertexBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uiVertexBufferMemories.resize(MAX_FRAMES_IN_FLIGHT);
        uiVertexBufferMapped.resize(MAX_FRAMES_IN_FLIGHT);
        uiVertices.reserve(MAX_UI_VERTICES);

        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            createBuffer(
                bufferSize,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                uiVertexBuffers[i],
                uiVertexBufferMemories[i]
            );

            if(vkMapMemory(
                    device,
                    uiVertexBufferMemories[i],
                    0,
                    bufferSize,
                    0,
                    &uiVertexBufferMapped[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to map UI vertex buffer");
            }
        }
    }

    void addUiQuad(float x,
                   float y,
                   float width,
                   float height,
                   const glm::vec2& uv0,
                   const glm::vec2& uv1,
                   const glm::vec4& color) {
        if(width <= 0.0f || height <= 0.0f) {
            return;
        }

        float x1 = x + width;
        float y1 = y + height;
        const glm::vec3 unusedBarycentric{0.0f};

        uiVertices.push_back({{x,  y,  0.0f}, color, {uv0.x, uv0.y}, unusedBarycentric});
        uiVertices.push_back({{x1, y,  0.0f}, color, {uv1.x, uv0.y}, unusedBarycentric});
        uiVertices.push_back({{x1, y1, 0.0f}, color, {uv1.x, uv1.y}, unusedBarycentric});

        uiVertices.push_back({{x,  y,  0.0f}, color, {uv0.x, uv0.y}, unusedBarycentric});
        uiVertices.push_back({{x1, y1, 0.0f}, color, {uv1.x, uv1.y}, unusedBarycentric});
        uiVertices.push_back({{x,  y1, 0.0f}, color, {uv0.x, uv1.y}, unusedBarycentric});
    }

    void addUiRect(float x,
                   float y,
                   float width,
                   float height,
                   const glm::vec4& color) {
        addUiQuad(x, y, width, height, solidUv, solidUv, color);
    }

    void addUiText(std::string_view text,
                   float x,
                   float baseline,
                   const glm::vec4& color) {
        for(unsigned char character : text) {
            if(character >= glyphs.size()) {
                continue;
            }

            const Glyph& glyph = glyphs[character];
            if(!glyph.valid) {
                continue;
            }

            float glyphX = x + static_cast<float>(glyph.bearing.x);
            float glyphY = baseline - static_cast<float>(glyph.bearing.y);

            addUiQuad(
                glyphX,
                glyphY,
                static_cast<float>(glyph.size.x),
                static_cast<float>(glyph.size.y),
                glyph.uv0,
                glyph.uv1,
                color
            );

            x += glyph.advance;
        }
    }

    float rotationSpeedRatio() const {
        return std::clamp(
            (rotationSpeedDeg - MIN_ROTATION_SPEED) /
                (MAX_ROTATION_SPEED - MIN_ROTATION_SPEED),
            0.0f,
            1.0f
        );
    }

    void updateUiBuffer(uint32_t frame) {
        uiVertices.clear();

        glm::vec2 cursor = getFramebufferCursorPosition();
        bool modeHovered = modeButton.contains(cursor);
        bool sliderHovered = speedSliderHitArea.contains(cursor) ||
                             dragTarget == DragTarget::RotationSpeed;


        char fpsText[32];
        std::snprintf(fpsText, sizeof(fpsText), "FPS: %.0f", displayedFps);
        addUiText(fpsText, 20.0f, 48.0f,
                  {0.35f, 1.0f, 0.62f, 1.0f});

        addUiText(
            wireframe ? "MODE: WIREFRAME" : "MODE: FILLED",
            modeButton.x ,
            modeButton.y + 26.0f,
            {0.92f, 0.98f, 1.0f, 1.0f}
        );

        char speedText[48];
        std::snprintf(
            speedText,
            sizeof(speedText),
            "ROTATION: %.0f DEG/S",
            rotationSpeedDeg
        );
        addUiText(speedText, 20.0f, 100.0f,
                  {0.82f, 0.90f, 1.0f, 1.0f});

        float ratio = rotationSpeedRatio();
        float filledWidth = speedSliderTrack.width * ratio;

        addUiRect(speedSliderTrack.x,
                  speedSliderTrack.y,
                  speedSliderTrack.width,
                  speedSliderTrack.height,
                  {0.13f, 0.17f, 0.22f, 1.0f});

        addUiRect(speedSliderTrack.x,
                  speedSliderTrack.y,
                  filledWidth,
                  speedSliderTrack.height,
                  sliderHovered
                      ? glm::vec4(0.35f, 0.90f, 1.0f, 1.0f)
                      : glm::vec4(0.20f, 0.70f, 1.0f, 1.0f));

        float knobX = speedSliderTrack.x + filledWidth;
        addUiRect(knobX - 4.0f,
                  speedSliderTrack.y - 5.0f,
                  8.0f,
                  16.0f,
                  {0.95f, 0.98f, 1.0f, 1.0f});

        if(uiVertices.size() > MAX_UI_VERTICES) {
            throw std::runtime_error("UI vertex buffer is too small");
        }

        std::memcpy(
            uiVertexBufferMapped[frame],
            uiVertices.data(),
            uiVertices.size() * sizeof(Vertex)
        );

        uiVertexCounts[frame] = static_cast<uint32_t>(uiVertices.size());
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 2);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createDescriptorSets() {
        // 首先要为每个 descriptor set指定layout
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();
        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorImageInfo textureInfo{};
            textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            textureInfo.imageView = textureImageView;
            textureInfo.sampler = textureSampler;

            VkDescriptorImageInfo fontInfo{};
            fontInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            fontInfo.imageView = fontImageView;
            fontInfo.sampler = fontSampler;

            std::array<VkWriteDescriptorSet, 3> descriptorWrites{};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];//目标是descriptorSet
            descriptorWrites[0].dstBinding = 0;//对应shader中的bind=0
            descriptorWrites[0].dstArrayElement = 0;//描述符可以是数组,默认0
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &textureInfo;

            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = descriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pImageInfo = &fontInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    void cleanup() {
        cleanupSwapChain();
        destroyGraphicsPipelines();
        vkDestroyRenderPass(device, renderPass, nullptr);

        for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            vkUnmapMemory(device, uiVertexBufferMemories[i]);
            vkDestroyBuffer(device, uiVertexBuffers[i], nullptr);
            vkFreeMemory(device, uiVertexBufferMemories[i], nullptr);

            vkUnmapMemory(device, uniformBuffersMemory[i]);
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);

            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        vkDestroySampler(device, fontSampler, nullptr);
        vkDestroyImageView(device, fontImageView, nullptr);
        vkDestroyImage(device, fontImage, nullptr);
        vkFreeMemory(device, fontImageMemory, nullptr);

        vkDestroySampler(device, textureSampler, nullptr);
        vkDestroyImageView(device, textureImageView, nullptr);
        vkDestroyImage(device, textureImage, nullptr);
        vkFreeMemory(device, textureImageMemory, nullptr);

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);
        if(enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void createTextureImage() {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

        if(!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
        VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth) * texHeight * 4;

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);

        stbi_image_free(pixels);

        createImage(texWidth, texHeight,mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

        generateMipmaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);
    }

    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = mipLevels;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples = numSamples;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;
        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommand(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;

        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = 0;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if(VK_IMAGE_LAYOUT_UNDEFINED == oldLayout && VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL == newLayout) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL == oldLayout && VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL == newLayout) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage, destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        endSingleTimeCommand(commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {
            width,
            height,
            1
        };

        vkCmdCopyBufferToImage(
            commandBuffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );

        endSingleTimeCommand(commandBuffer);
    }

    void createTextureImageView() {
        textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
    }

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlagBits aspectFlags, uint32_t mipLevels) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if(VK_SUCCESS != vkCreateImageView(device, &viewInfo, nullptr, &imageView)) {
            throw std::runtime_error("failed to create texture image view!");
        }
        return imageView;
    }

    void createTextureSampler() {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_FALSE;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
        if(VK_SUCCESS != vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler)) {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    void createFontAtlas() {
        std::vector<uint8_t> pixels(
            static_cast<size_t>(FONT_ATLAS_WIDTH) * FONT_ATLAS_HEIGHT,
            0
        );

        // UI 矩形会采样这个纯白像素，所以按钮和滑块不需要另一张纹理。
        pixels[0] = 255;

        FT_Library library = nullptr;
        FT_Face face = nullptr;

        if(FT_Init_FreeType(&library) != 0) {
            throw std::runtime_error("FT_Init_FreeType failed");
        }

        if(FT_New_Face(library, FONT_PATH.c_str(), 0, &face) != 0) {
            FT_Done_FreeType(library);
            throw std::runtime_error("failed to load OTF font: " + FONT_PATH);
        }

        FT_Select_Charmap(face, FT_ENCODING_UNICODE);

        if(FT_Set_Pixel_Sizes(face, 0, FONT_PIXEL_SIZE) != 0) {
            FT_Done_Face(face);
            FT_Done_FreeType(library);
            throw std::runtime_error("FT_Set_Pixel_Sizes failed");
        }

        int penX = 2;
        int penY = 1;
        int rowHeight = 0;

        for(uint32_t character = 32; character < 127; ++character) {
            if(FT_Load_Char(
                    face,
                    character,
                    FT_LOAD_RENDER | FT_LOAD_TARGET_NORMAL) != 0) {
                continue;
            }

            FT_GlyphSlot slot = face->glyph;
            FT_Bitmap& bitmap = slot->bitmap;

            if(penX + static_cast<int>(bitmap.width) + 1 >=
               static_cast<int>(FONT_ATLAS_WIDTH)) {
                penX = 1;
                penY += rowHeight + 1;
                rowHeight = 0;
            }

            if(penY + static_cast<int>(bitmap.rows) + 1 >=
               static_cast<int>(FONT_ATLAS_HEIGHT)) {
                FT_Done_Face(face);
                FT_Done_FreeType(library);
                throw std::runtime_error("font atlas is too small");
            }

            for(uint32_t row = 0; row < bitmap.rows; ++row) {
                const uint8_t* sourceRow = nullptr;

                if(bitmap.pitch >= 0) {
                    sourceRow = bitmap.buffer + row * bitmap.pitch;
                } else {
                    sourceRow = bitmap.buffer +
                        (bitmap.rows - 1 - row) * static_cast<uint32_t>(-bitmap.pitch);
                }

                std::memcpy(
                    pixels.data() +
                        static_cast<size_t>(penY + static_cast<int>(row)) * FONT_ATLAS_WIDTH +
                        penX,
                    sourceRow,
                    bitmap.width
                );
            }

            Glyph& glyph = glyphs[character];
            glyph.size = {
                static_cast<int>(bitmap.width),
                static_cast<int>(bitmap.rows)
            };
            glyph.bearing = {slot->bitmap_left, slot->bitmap_top};
            glyph.advance = static_cast<float>(slot->advance.x) / 64.0f;
            glyph.uv0 = {
                static_cast<float>(penX) / FONT_ATLAS_WIDTH,
                static_cast<float>(penY) / FONT_ATLAS_HEIGHT
            };
            glyph.uv1 = {
                static_cast<float>(penX + static_cast<int>(bitmap.width)) /
                    FONT_ATLAS_WIDTH,
                static_cast<float>(penY + static_cast<int>(bitmap.rows)) /
                    FONT_ATLAS_HEIGHT
            };
            glyph.valid = true;

            penX += static_cast<int>(bitmap.width) + 1;
            rowHeight = std::max(rowHeight, static_cast<int>(bitmap.rows));
        }

        FT_Done_Face(face);
        FT_Done_FreeType(library);

        VkDeviceSize imageSize = pixels.size();
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;

        createBuffer(
            imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingMemory
        );

        void* mapped = nullptr;
        if(vkMapMemory(device, stagingMemory, 0, imageSize, 0, &mapped) != VK_SUCCESS) {
            throw std::runtime_error("failed to map font staging buffer");
        }
        std::memcpy(mapped, pixels.data(), pixels.size());
        vkUnmapMemory(device, stagingMemory);

        createImage(
            FONT_ATLAS_WIDTH,
            FONT_ATLAS_HEIGHT,
            1,
            VK_SAMPLE_COUNT_1_BIT,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            fontImage,
            fontImageMemory
        );

        transitionImageLayout(
            fontImage,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1
        );

        copyBufferToImage(
            stagingBuffer,
            fontImage,
            FONT_ATLAS_WIDTH,
            FONT_ATLAS_HEIGHT
        );

        transitionImageLayout(
            fontImage,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            1
        );

        fontImageView = createImageView(
            fontImage,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_ASPECT_COLOR_BIT,
            1
        );

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
    }

    void createFontSampler() {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_WHITE;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        if(vkCreateSampler(device, &samplerInfo, nullptr, &fontSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create font sampler");
        }
    }

    void createDepthResources() {
        VkFormat depthFormat = findDepthFormat();
        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
    }

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
        for(VkFormat format : candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
            if(tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            } else if(tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }
        throw std::runtime_error("failed to find supported format!");
    }

    VkFormat findDepthFormat() {
        return findSupportedFormat({VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT}, 
                VK_IMAGE_TILING_OPTIMAL, 
                VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    void loadModel() {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warning;
        std::string err;
        if(!tinyobj::LoadObj(
                &attrib,
                &shapes,
                &materials,
                &warning,
                &err,
                MODEL_PATH.c_str(),
                nullptr,
                true)) {
            throw std::runtime_error(err);
        }
        for(const auto& shape : shapes) {
            for(const auto&index : shape.mesh.indices) {
                Vertex vertex{};
                vertex.pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2],
                };
                if(index.texcoord_index >= 0) {
                    vertex.texCoord = {
                        attrib.texcoords[2 * index.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                    };
                } else {
                    vertex.texCoord = {0.0f, 0.0f};
                }

                uint32_t corner = static_cast<uint32_t>(indices.size() % 3);
                if(corner == 0) {
                    vertex.barycentric = {1.0f, 0.0f, 0.0f};
                } else if(corner == 1) {
                    vertex.barycentric = {0.0f, 1.0f, 0.0f};
                } else {
                    vertex.barycentric = {0.0f, 0.0f, 1.0f};
                }

                vertices.push_back(vertex);
                indices.push_back(indices.size());
            }
        }
    }

    void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {

        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

        if(!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
            throw std::runtime_error("texture image format does not support linear blitting!");
        }

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;

        int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;

        for(uint32_t i = 1; i < mipLevels; i++) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 
                    0, nullptr,
                    0, nullptr,
                    1, &barrier);
            VkImageBlit blit{};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;
            vkCmdBlitImage(commandBuffer,
                    image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &blit,
                    VK_FILTER_LINEAR);
            if(mipWidth > 1) mipWidth /= 2;
            if(mipHeight > 1) mipHeight /= 2;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

        }

        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

        endSingleTimeCommand(commandBuffer);
    }

    VkSampleCountFlagBits getMaxUsableSampleCount() {
        VkPhysicalDeviceProperties physicalDeviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

        VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
        if(counts & VK_SAMPLE_COUNT_64_BIT) {return VK_SAMPLE_COUNT_64_BIT;}
        if(counts & VK_SAMPLE_COUNT_32_BIT) {return VK_SAMPLE_COUNT_32_BIT;}
        if(counts & VK_SAMPLE_COUNT_16_BIT) {return VK_SAMPLE_COUNT_16_BIT;}
        if(counts & VK_SAMPLE_COUNT_8_BIT) {return VK_SAMPLE_COUNT_8_BIT;}
        if(counts & VK_SAMPLE_COUNT_4_BIT) {return VK_SAMPLE_COUNT_4_BIT;}
        if(counts & VK_SAMPLE_COUNT_2_BIT) {return VK_SAMPLE_COUNT_2_BIT;}
        return VK_SAMPLE_COUNT_1_BIT;
    }

    void createColorResources() {
        VkFormat colorFormat = swapChainImageFormat;
        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
        colorImageView = createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }

    uint32_t buildBVHNode(uint32_t first, uint32_t count) {
        const uint32_t nodeIndex = static_cast<uint32_t>(bvhNodes.size());
        bvhNodes.emplace_back();

        AABB nodeBounds;
        AABB centroidBounds;

        for(uint32_t i = first; i < first + count; ++i) {
            nodeBounds.expand(bvhPrimitives[i].bounds);
            centroidBounds.expand(bvhPrimitives[i].centroid);
        }
        bvhNodes[nodeIndex].bounds = nodeBounds;

        constexpr uint32_t LEAF_TRIANGLE_COUNT = 8;

        glm::vec3 extent = centroidBounds.hi - centroidBounds.lo;

        int splitAxis = 0;
        if(extent.y > extent.x) {
            splitAxis = 1;
        }
        if(extent.z > extent[splitAxis]) {
            splitAxis = 2;
        }

        if(count <= LEAF_TRIANGLE_COUNT || extent[splitAxis] < 1e-6f) {
           bvhNodes[nodeIndex].first = first;
           bvhNodes[nodeIndex].count = count;
           return nodeIndex;
        }

        const uint32_t middle = first + count / 2;

        std::nth_element(
            bvhPrimitives.begin() + first,
            bvhPrimitives.begin() + middle,
            bvhPrimitives.begin() + first + count,
            [splitAxis](const BVHPrimitive& a,
                const BVHPrimitive& b) {
                return a.centroid[splitAxis] < b.centroid[splitAxis];
            }
        );

        const uint32_t left = buildBVHNode(first, middle - first);
        const uint32_t right = buildBVHNode(middle, first + count - middle);

        bvhNodes[nodeIndex].left = left;
        bvhNodes[nodeIndex].right = right;

        return nodeIndex;
    }

    void buildBVH() {
        if(indices.empty() || indices.size() % 3 != 0) {
            throw std::runtime_error("model does not contain valid");
        }
        bvhPrimitives.clear();
        bvhNodes.clear();

        const uint32_t triangleCount = static_cast<uint32_t>(indices.size() / 3);

        bvhPrimitives.reserve(triangleCount);
        bvhNodes.reserve(triangleCount * 2);

        for(uint32_t triangle = 0; triangle < triangleCount; ++triangle) {
            const glm::vec3& p0 = vertices[indices[triangle * 3 + 0]].pos;
            const glm::vec3& p1 = vertices[indices[triangle * 3 + 1]].pos;
            const glm::vec3& p2 = vertices[indices[triangle * 3 + 2]].pos;

            BVHPrimitive primitive{};
            primitive.triangle = triangle;
            primitive.bounds.expand(p0);
            primitive.bounds.expand(p1);
            primitive.bounds.expand(p2);
            primitive.centroid = (p0 + p1 + p2) / 3.0f;

            bvhPrimitives.push_back(primitive);
        }

        const uint32_t root = buildBVHNode(0, triangleCount);
        const AABB& modelBounds = bvhNodes[root].bounds;
        modelPivot = (modelBounds.lo + modelBounds.hi) * 0.5f;
        modelRadius = glm::length(modelBounds.hi - modelBounds.lo) * 0.5f;
        modelRadius = std::max(modelRadius, 0.001f);
        cameraDistance = modelRadius * 3.0f;
        minCameraDistance = modelRadius * 0.3f;
        maxCameraDistance = modelRadius * 100.0f;
    }

    glm::mat4 getModelMatrix() const {
        glm::quat automaticRotation = glm::angleAxis(
            automaticRotationRadians,
            glm::vec3(0.0f, 0.0f, 1.0f)
        );

        glm::quat combinedRotation = glm::normalize(
            modelRotation * automaticRotation
        );

        return
            glm::translate(glm::mat4(1.0f), modelPivot) *
            glm::mat4_cast(combinedRotation) *
            glm::translate(glm::mat4(1.0f), -modelPivot
        );
    }

    glm::mat4 getViewMatrix() const {
        glm::vec3 cameraPosition = modelPivot + cameraDirection * cameraDistance;
        return glm::lookAt(
            cameraPosition,
            modelPivot, 
            glm::vec3(0.0f, 0.0f, 1.0f)
        );
    }

    glm::mat4 getProjectionMatrix() const {
        float aspect = 
            static_cast<float>(swapChainExtent.width) / 
            static_cast<float>(std::max(1u, swapChainExtent.height)
        );

        float zNear = std::max(
            0.001f,
            cameraDistance - modelRadius * 2.0f
        );

        float zFar = std::max(
            zNear + 1.0f,
            cameraDistance + modelRadius * 2.0f
        );

        glm::mat4 projection = glm::perspective(
            glm::radians(45.0f),
            aspect,
            zNear,
            zFar
        );

        projection[1][1] *= -1.0f;

        return projection;
    }

    Ray makeWorldRay(double cursorX, double cursorY) const {
        int windowWidth = 1;
        int windowHeight = 1;
        int framebufferWidth = 1;
        int framebufferHeight = 1;
        glfwGetWindowSize(
            window,
            &windowWidth,
            &windowHeight
        );

        glfwGetFramebufferSize(
            window,
            &framebufferWidth,
            &framebufferHeight
        );

        double pixelX = cursorX * framebufferWidth / windowWidth;
        double pixelY = cursorY * framebufferHeight / windowHeight;
        float ndcX = 2.0f * static_cast<float>(pixelX) / framebufferWidth - 1.0f;
        float ndcY = 2.0f * static_cast<float>(pixelY) / framebufferHeight - 1.0f;
        glm::mat4 inversePV = glm::inverse(
            getProjectionMatrix() * getViewMatrix()
        );
        glm::vec4 nearPosition = inversePV * glm::vec4(ndcX, ndcY, 0.0f, 1.0f);
        glm::vec4 farPosition = inversePV * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
        nearPosition /= nearPosition.w;
        farPosition /= farPosition.w;
        Ray ray{};
        ray.origin = glm::vec3(nearPosition);
        ray.direction = glm::normalize(glm::vec3(farPosition - nearPosition));
        return ray;
    }

    bool intersectAABB(
        const Ray& ray,
        const AABB& box,
        float maximumDistance
    ) const {
        float tMin = 0.0f;
        float tMax = maximumDistance;
        for (int axis = 0; axis < 3; ++axis) {
            float origin = ray.origin[axis];
            float direction = ray.direction[axis];
            if (std::abs(direction) < 1e-8f) {
                if (origin < box.lo[axis] ||
                    origin > box.hi[axis]) {
                    return false;
                }
                continue;
            }
            float t0 = (box.lo[axis] - origin) / direction;
            float t1 = (box.hi[axis] - origin) / direction;
            if (t0 > t1) {
                std::swap(t0, t1);
            }
            tMin = std::max(tMin, t0);
            tMax = std::min(tMax, t1);
            if (tMin > tMax) {
                return false;
            }
        }
        return true;
    }

    bool intersectTriangle(
        const Ray& ray,
        uint32_t triangle,
        float& resultDistance
    ) const {
        const glm::vec3& p0 = vertices[indices[triangle * 3 + 0]].pos;
        const glm::vec3& p1 = vertices[indices[triangle * 3 + 1]].pos;
        const glm::vec3& p2 = vertices[indices[triangle * 3 + 2]].pos;

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;

        glm::vec3 p = glm::cross(ray.direction, edge2);
        float determinant = glm::dot(edge1, p);

        if (std::abs(determinant) < 1e-8f) {
            return false;
        }

        float inverseDeterminant = 1.0f / determinant;

        glm::vec3 t = ray.origin - p0;
        float u = glm::dot(t, p) * inverseDeterminant;

        if (u < 0.0f || u > 1.0f) {
            return false;
        }
        glm::vec3 q = glm::cross(t, edge1);
        float v = glm::dot(ray.direction, q) * inverseDeterminant;
        if (v < 0.0f || u + v > 1.0f) {
            return false;
        }
        float distance =
            glm::dot(edge2, q) * inverseDeterminant;
        if (distance <= 1e-5f) {
            return false;
        }
        resultDistance = distance;
        return true;
    }

    PickHit pickModel(double cursorX, double cursorY) const {
        PickHit result{};

        if (bvhNodes.empty()) {
            return result;
        }

        Ray worldRay = makeWorldRay(cursorX, cursorY);

        glm::mat4 inverseModel =
        glm::inverse(getModelMatrix());

        Ray localRay{};
        localRay.origin = glm::vec3(inverseModel * glm::vec4(worldRay.origin, 1.0f));

        localRay.direction = glm::vec3(inverseModel * glm::vec4(worldRay.direction, 0.0f));

        std::vector<uint32_t> stack;
        stack.push_back(0);

        while (!stack.empty()) {
            uint32_t nodeIndex = stack.back();
            stack.pop_back();
        
            const BVHNode& node = bvhNodes[nodeIndex];
        
            if (!intersectAABB(localRay, node.bounds, result.distance)) {
                continue;
            }
        
            if (node.isLeaf()) {
                for (uint32_t i = 0; i < node.count; ++i) {
                    const BVHPrimitive& primitive = bvhPrimitives[node.first + i];

                    float distance = 0.0f;

                    if (intersectTriangle(
                            localRay,
                            primitive.triangle,
                            distance) &&
                        distance < result.distance) {
                        result.hit = true;
                        result.distance = distance;
                        result.triangle =
                            primitive.triangle;
                    }
                }
            } else {
                stack.push_back(node.left);
                stack.push_back(node.right);
            }
        }

        if (result.hit) {
            result.worldPosition = worldRay.origin + worldRay.direction * result.distance;
        }

        return result;
    }

    glm::vec2 windowToFramebuffer(double cursorX, double cursorY) const {
        int windowWidth = 1;
        int windowHeight = 1;
        glfwGetWindowSize(window, &windowWidth, &windowHeight);

        return {
            static_cast<float>(
                cursorX * swapChainExtent.width /
                static_cast<double>(std::max(windowWidth, 1))
            ),
            static_cast<float>(
                cursorY * swapChainExtent.height /
                static_cast<double>(std::max(windowHeight, 1))
            )
        };
    }

    glm::vec2 getFramebufferCursorPosition() const {
        double cursorX = 0.0;
        double cursorY = 0.0;
        glfwGetCursorPos(window, &cursorX, &cursorY);
        return windowToFramebuffer(cursorX, cursorY);
    }

    void updateRotationSpeed(const glm::vec2& cursor) {
        float ratio =
            (cursor.x - speedSliderTrack.x) /
            speedSliderTrack.width;

        ratio = std::clamp(ratio, 0.0f, 1.0f);
        rotationSpeedDeg = MIN_ROTATION_SPEED +
            ratio * (MAX_ROTATION_SPEED - MIN_ROTATION_SPEED);
    }

    glm::vec3 mapToArcball(double cursorX, double cursorY) const {
        int width = 1;
        int height = 1;
        glfwGetWindowSize(window, &width, &height);

        float size = static_cast<float>(
            std::max(1, std::min(width, height))
        );

        float x = (static_cast<float>(cursorX) - static_cast<float>(width) * 0.5f ) / (size * 0.5f);
	    float y = (static_cast<float>(height)  * 0.5f - static_cast<float>(cursorY)) / (size * 0.5f);

        float lengthSquared = x * x + y * y;

        if (lengthSquared <= 1.0f) {
            return glm::vec3(x, y, std::sqrt(1.0f - lengthSquared));
        }

        return glm::normalize(glm::vec3(x, y, 0.0f));
    }

    static void mouseButtonCallback(
        GLFWwindow* window,
        int button,
        int action,
        int mods
    ) {
        auto* app = reinterpret_cast<VulkanHut*>(
            glfwGetWindowUserPointer(window)
        );
        if (button != GLFW_MOUSE_BUTTON_LEFT) {
            return;
        }

        double cursorX = 0.0;
        double cursorY = 0.0;
        glfwGetCursorPos(window, &cursorX, &cursorY);
        glm::vec2 framebufferCursor = app->windowToFramebuffer(cursorX, cursorY);

        if (action == GLFW_PRESS) {
            if(app->modeButton.contains(framebufferCursor)) {
                app->wireframe = !app->wireframe;
                app->dragTarget = DragTarget::None;
                return;
            }

            if(app->speedSliderHitArea.contains(framebufferCursor)) {
                app->dragTarget = DragTarget::RotationSpeed;
                app->updateRotationSpeed(framebufferCursor);
                return;
            }

            PickHit hit = app->pickModel(cursorX, cursorY);

            if (hit.hit) {
                app->dragTarget = DragTarget::Model;
                app->selectedTriangle = hit.triangle;
                app->dragStartBall = app->mapToArcball(cursorX, cursorY);
                app->dragStartRotation = app->modelRotation;
            }
        }

        if (action == GLFW_RELEASE) {
            app->dragTarget = DragTarget::None;
        }
    }

    static void cursorPositionCallback(
        GLFWwindow* window,
        double cursorX,
        double cursorY
    ) {
        auto* app = reinterpret_cast<VulkanHut*>(
            glfwGetWindowUserPointer(window)
        );

        if(app->dragTarget == DragTarget::RotationSpeed) {
            app->updateRotationSpeed(
                app->windowToFramebuffer(cursorX, cursorY)
            );
            return;
        }

        if (app->dragTarget != DragTarget::Model) {
            return;
        }

        glm::vec3 currentBall = app->mapToArcball(cursorX, cursorY);
        glm::vec3 cameraAxis = glm::cross(app->dragStartBall, currentBall);

        float axisLength = glm::length(cameraAxis);
        if (axisLength < 1e-6f) {
            app->modelRotation =
                app->dragStartRotation;
            return;
        }
        cameraAxis /= axisLength;
        float angle = std::acos(std::clamp(
            glm::dot(app->dragStartBall, currentBall),
            -1.0f,
            1.0f
        ));
        glm::vec3 worldAxis = glm::normalize(
            glm::mat3(glm::inverse(app->getViewMatrix())) *
            cameraAxis
        );
        glm::quat deltaRotation = glm::angleAxis(angle, worldAxis);
        app->modelRotation = glm::normalize(deltaRotation * app->dragStartRotation);
    }

    static void scrollCallback(
        GLFWwindow* window,
        double xOffset,
        double yOffset
    ) {
        auto* app = reinterpret_cast<VulkanHut*>(
            glfwGetWindowUserPointer(window)
        );

        (void)xOffset;

        app->cameraDistance *= std::exp(
            -0.12f * static_cast<float>(yOffset)
        );

        app->cameraDistance = std::clamp(
            app->cameraDistance,
            app->minCameraDistance,
            app->maxCameraDistance
        );
    }
};

int main() {

    try {
        VulkanHut app;
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
