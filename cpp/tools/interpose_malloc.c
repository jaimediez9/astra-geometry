// LD_PRELOAD interposer that counts malloc-family calls between arm and disarm.
//
// PLATFORM: Linux with glibc. This is a HOST-SPECIFIC measurement, not a
// portable property of the code under test. It works because the dynamic linker
// resolves these symbols ahead of libc's, and it reaches the real allocator
// through glibc's internal entry points. On a different platform, a static
// link, or with a different libc, the harness is expected to report the layer as
// UNAVAILABLE -- never as "no allocations".
//
// Build:  cc -shared -fPIC -o astra_interpose_malloc.so interpose_malloc.c
// Use:    LD_PRELOAD=./astra_interpose_malloc.so ./memory_guards

#define _GNU_SOURCE

#include <stdatomic.h>
#include <stddef.h>

/* glibc's internal allocator entry points: using them avoids re-entering our
   own interposers while resolving symbols, which is the usual trap here. */
extern void* __libc_malloc(size_t size);
extern void* __libc_calloc(size_t count, size_t size);
extern void* __libc_realloc(void* pointer, size_t size);
extern void __libc_free(void* pointer);

static _Atomic long g_allocation_count = 0;
static _Atomic int g_armed = 0;

/* --- interface consumed by the harness through dlsym(RTLD_DEFAULT, ...) ----- */

void astra_malloc_guard_reset(void) { atomic_store(&g_allocation_count, 0); }

void astra_malloc_guard_arm(void) { atomic_store(&g_armed, 1); }

void astra_malloc_guard_disarm(void) { atomic_store(&g_armed, 0); }

long astra_malloc_guard_count(void) { return atomic_load(&g_allocation_count); }

const char* astra_malloc_guard_platform(void) { return "linux-glibc/LD_PRELOAD"; }

/* --- interposed allocator ------------------------------------------------- */

static void note_allocation(void) {
  if (atomic_load(&g_armed)) {
    atomic_fetch_add(&g_allocation_count, 1);
  }
}

void* malloc(size_t size) {
  note_allocation();
  return __libc_malloc(size);
}

void* calloc(size_t count, size_t size) {
  note_allocation();
  return __libc_calloc(count, size);
}

void* realloc(void* pointer, size_t size) {
  note_allocation();
  return __libc_realloc(pointer, size);
}

void free(void* pointer) { __libc_free(pointer); }
