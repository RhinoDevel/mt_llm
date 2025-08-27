
// Marcel Timm, RhinoDevel, 2024aug21

#ifndef MT_LLM_LIB
#define MT_LLM_LIB

#ifdef _WIN32
	#ifdef MT_EXPORT_LLM
		#define MT_EXPORT_LLM_API __declspec(dllexport)
	#else
		#define MT_EXPORT_LLM_API __declspec(dllimport)
	#endif //MT_EXPORT_LLM
#else //_WIN32
	#define MT_EXPORT_LLM_API
	#ifndef __stdcall
		#define __stdcall
	#endif //__stdcall
#endif //_WIN32

#endif //MT_LLM_LIB
