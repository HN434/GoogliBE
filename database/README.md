# Database Setup for Commentary System

## PostgreSQL Setup

### 1. Install PostgreSQL

**Windows:**
- Download from https://www.postgresql.org/download/windows/
- Or use: `choco install postgresql`

**Linux:**
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
```

**Mac:**
```bash
brew install postgresql
```

### 2. Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE cricket_commentary;

# Create user (optional)
CREATE USER cricket_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE cricket_commentary TO cricket_user;

# Exit
\q
```

### 3. Configure Environment Variables

Add to `.env` file:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=cricket_commentary
DB_ECHO=false
```

### 4. Run Migrations

**Option 1: Using SQL Script**
```bash
psql -U postgres -d cricket_commentary -f database/migrations/001_initial_schema.sql
```

**Option 2: Automatic (via SQLAlchemy)**
The tables will be created automatically when the application starts if they don't exist.

### 5. Verify Setup

```bash
# Connect to database
psql -U postgres -d cricket_commentary

# Check tables
\dt

# Check matches table structure
\d matches

# Check commentaries table structure
\d commentaries

# Exit
\q
```

## Database Schema

### Matches Table
- Stores match metadata and status
- Indexed on `match_id`, `state`, `is_complete`
- Tracks match start/end timestamps

### Commentaries Table
- Stores individual commentary lines
- Foreign key to matches table
- Indexed on `match_id`, `timestamp`, `event_type`, `over_number`, `ball_number`
- Supports efficient queries by match, time, and event type

## Usage

The database service is automatically integrated into the commentary worker. Commentaries are stored:
- When fetched from the API
- Before being published to Redis
- With match status updates

## API Endpoints

The database can be accessed via:
- Direct database queries
- Future REST API endpoints (to be added)

## Backup

```bash
# Backup database
pg_dump -U postgres cricket_commentary > backup.sql

# Restore database
psql -U postgres cricket_commentary < backup.sql
```

