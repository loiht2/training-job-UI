import { NextResponse } from "next/server";
import { mkdir, readFile, rm, writeFile } from "fs/promises";
import path from "path";

type JobStatus = "Pending" | "Running" | "Succeeded" | "Failed";

type StoredJob = {
  id: string;
  algorithm: string;
  createdAt: number;
  priority: number;
  status: JobStatus;
  pendingUntil?: number;
};

const JOB_OUTPUT_DIR = path.join(process.cwd(), "tmp", "jobs");
const INDEX_PATH = path.join(JOB_OUTPUT_DIR, "index.json");
const JOB_STATUSES = new Set<JobStatus>(["Pending", "Running", "Succeeded", "Failed"]);

async function readIndex(): Promise<StoredJob[]> {
  try {
    const raw = await readFile(INDEX_PATH, "utf8");
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((item) => {
        if (!item || typeof item !== 'object') return null;
        const maybe = item as Partial<StoredJob>;
        if (
          typeof maybe.id !== 'string' ||
          typeof maybe.algorithm !== 'string' ||
          typeof maybe.createdAt !== 'number' ||
          typeof maybe.priority !== 'number' ||
          typeof maybe.status !== 'string' ||
          !JOB_STATUSES.has(maybe.status as JobStatus)
        ) {
          return null;
        }
        const entry: StoredJob = {
          id: maybe.id,
          algorithm: maybe.algorithm,
          createdAt: maybe.createdAt,
          priority: maybe.priority,
          status: maybe.status as JobStatus,
        };
        if (typeof maybe.pendingUntil === 'number') {
          entry.pendingUntil = maybe.pendingUntil;
        }
        return entry;
      })
      .filter((item): item is StoredJob => Boolean(item));
  } catch {
    return [];
  }
}

async function writeIndex(entries: StoredJob[]) {
  await mkdir(JOB_OUTPUT_DIR, { recursive: true });
  await writeFile(INDEX_PATH, JSON.stringify(entries, null, 2), 'utf8');
}

export async function GET() {
  try {
    const jobs = await readIndex();
    const now = Date.now();
    let mutated = false;
    const normalized = jobs.map((job) => {
      if (job.status === 'Pending' && job.pendingUntil && now >= job.pendingUntil) {
        mutated = true;
        return { ...job, status: 'Running' as JobStatus, pendingUntil: undefined };
      }
      return job;
    });
    if (mutated) {
      await writeIndex(normalized);
    }
    return NextResponse.json({ ok: true, jobs: normalized });
  } catch (error) {
    console.error('Failed to read job index', error);
    return NextResponse.json({ ok: false, jobs: [] }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as { payload?: unknown; job?: unknown };
    if (!body || typeof body !== 'object') {
      return NextResponse.json({ ok: false, message: 'Invalid payload' }, { status: 400 });
    }

    const payload = body.payload;
    const jobData = body.job;

    if (!payload || typeof payload !== 'object' || !jobData || typeof jobData !== 'object') {
      return NextResponse.json({ ok: false, message: 'Invalid payload structure' }, { status: 400 });
    }

    const jobName = typeof (payload as { jobName?: unknown }).jobName === 'string' ? (payload as { jobName: string }).jobName : 'job';
    const safeName = jobName.replace(/[^a-z0-9-_]/gi, '-').toLowerCase();
    const filename = `${safeName}-${Date.now()}.json`;

    const rawJob = jobData as Partial<StoredJob>;
    if (
      typeof rawJob.id !== 'string' ||
      typeof rawJob.algorithm !== 'string' ||
      typeof rawJob.createdAt !== 'number' ||
      typeof rawJob.priority !== 'number' ||
      typeof rawJob.status !== 'string' ||
      !JOB_STATUSES.has(rawJob.status as JobStatus)
    ) {
      return NextResponse.json({ ok: false, message: 'Invalid job metadata' }, { status: 400 });
    }

    const storedJob: StoredJob = {
      id: rawJob.id,
      algorithm: rawJob.algorithm,
      createdAt: rawJob.createdAt,
      priority: rawJob.priority,
      status: rawJob.status as JobStatus,
    };
    if (typeof rawJob.pendingUntil === 'number') {
      storedJob.pendingUntil = rawJob.pendingUntil;
    }

    await mkdir(JOB_OUTPUT_DIR, { recursive: true });
    await writeFile(path.join(JOB_OUTPUT_DIR, filename), JSON.stringify(payload, null, 2), 'utf8');

    const existing = await readIndex();
    const filtered = existing.filter((item) => item.id !== storedJob.id);
    filtered.push(storedJob);
    await writeIndex(filtered);

    return NextResponse.json({ ok: true, filename, job: storedJob });
  } catch (error) {
    console.error('Failed to persist job payload', error);
    return NextResponse.json({ ok: false, message: 'Failed to save job payload' }, { status: 500 });
  }
}

export async function DELETE() {
  try {
    await rm(JOB_OUTPUT_DIR, { recursive: true, force: true });
    return NextResponse.json({ ok: true });
  } catch (error) {
    console.error('Failed to clear job payloads', error);
    return NextResponse.json({ ok: false, message: 'Failed to clear saved payloads' }, { status: 500 });
  }
}
